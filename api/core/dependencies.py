"""
UCMS API — core/dependencies.py
Shared data access helpers.
"""

from fastapi import Request
import pandas as pd


def get_master(request: Request) -> pd.DataFrame:
    return request.app.state.master


def get_summary(request: Request) -> pd.DataFrame:
    return request.app.state.summary


def get_weights(request: Request) -> pd.DataFrame:
    return request.app.state.weights