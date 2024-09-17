"""
This script shows how to initialize, start, and stop the hardware monitor. 
"""
import threading

from .monitor import HWMonitor


class Client(object): 
  
  def __init__(): 
    # This automatically starts the hardware monitor when a job is launched.
    self.monitor = HWMonitor(monitoring_freq=1, stop_event=threading.Event())
    
  def stop_monitor(): 
    # This stops the asynchronous hardware monitor. 
    self.monitor.stop_monitor.set()