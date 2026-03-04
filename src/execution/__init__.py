"""
Execution module for native exchange order management.

Provides LiveOrderManager for placing native SL/TP/trailing stop orders
on HyperLiquid instead of relying on polling-based position monitoring.
"""

from .order_manager import LiveOrderManager

__all__ = ['LiveOrderManager']
