#!/usr/bin/env python3
"""Advanced trading environment supporting shorts, slippage and commissions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import gymnasium as gym  # Usamos gymnasium en lugar de gym
import numpy as np
import pandas as pd

from .rewards import REWARD_REGISTRY, RewardCalculator


@dataclass
class EnvironmentConfig:
    """Configuración para el TradingEnvironment."""
    data: pd.DataFrame
    initial_cash: float = 1_000_000
    commission_rate: float = 0.001
    slippage: float = 0.0005
    use_continuous_action: bool = False
    reward: str = "pnl"


class TradingEnvironment(gym.Env):
    """Trading simulator that supports shorting and configurable rewards."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnvironmentConfig):
        super().__init__()
        if config.data is None or config.data.empty:
            raise ValueError("Environment requires a non-empty DataFrame of features")
        self.config = config
        
        # Guardar la columna de fecha para referencia
        self.dates = config.data.get("date", pd.RangeIndex(start=0, stop=len(config.data)))
        
        # Asegurar que 'close' exista antes de intentar eliminarlo
        if "close" not in config.data.columns:
             raise KeyError("La columna 'close' es requerida en los datos para precios.")
        self.prices = config.data["close"].values
        
        # Preparar features: eliminar columnas no numéricas/no deseadas
        feature_df = config.data.drop(columns=["close", "date"], errors="ignore")
        
        # Convertir tipos de datos para el vector de observación
        for column in feature_df.select_dtypes(include=["datetime", "datetimetz"]).columns:
            feature_df[column] = feature_df[column].astype("int64")
        for column in feature_df.select_dtypes(include=["object"]).columns:
            feature_df[column] = feature_df[column].astype("category").cat.codes
            
        self.features = feature_df.astype(np.float32).values
        
        self.reward_calculator: RewardCalculator = self._build_reward_calculator(config)

        # Definición de Espacios
        if config.use_continuous_action:
            # Acción = % de cartera a mantener (de -100% short a +100% long)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            # 0 = Short, 1 = Flat, 2 = Long
            self.action_space = gym.spaces.Discrete(3)

        feature_dim = self.features.shape[1]
        # Estado = features + 3 métricas de cartera (cash, position_shares, portfolio_value)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_dim + 3,), dtype=np.float32
        )

        # Inicializar estado interno
        self.current_step = 0
        self.cash = 0.0
        self.position = 0.0
        self.position_value = 0.0
        self.portfolio_value = 0.0
        self.trades = [] # Historial de trades

    # ------------------------------------------------------------------
    # API de Gymnasium
    # ------------------------------------------------------------------
    
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reinicia el entorno al estado inicial."""
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.config.initial_cash
        self.position = 0.0  # (positivo long, negativo short, en número de acciones)
        self.position_value = 0.0
        self.portfolio_value = self.cash
        self.reward_calculator.reset(self.portfolio_value)
        self.trades = []
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Ejecuta un paso en el entorno."""
        
        if self.config.use_continuous_action:
            target_ratio = float(np.clip(action[0], -1.0, 1.0))
        else:
            # 0 = Short (-1), 1 = Flat (0), 2 = Long (1)
            target_ratio = float(int(action) - 1.0) 

        price = self.prices[self.current_step]
        
        # Calcular el valor nocional objetivo
        target_notional = target_ratio * self.portfolio_value
        
        # Calcular el número de acciones objetivo
        target_position = target_notional / price if price > 0 else 0.0
        
        # Calcular la diferencia (delta) de acciones a operar
        delta = target_position - self.position
        
        # Simular slippage en el precio de ejecución
        executed_price = price * (1 + self.config.slippage * np.sign(delta)) if delta != 0 else price
        
        # Calcular costos
        trade_cost_notional = executed_price * delta
        commission = abs(trade_cost_notional) * self.config.commission_rate

        # Actualizar cartera
        self.cash -= (trade_cost_notional + commission)
        self.position = target_position
        self.position_value = self.position * price # Valor de la posición al precio de mercado
        
        # El valor de la cartera es el efectivo más el valor de la posición
        self.portfolio_value = self.cash + self