#!/usr/bin/env python3
"""
Stepper control for A4988 using lgpio on Raspberry Pi 5
Safe slow pulses with debug output
"""

import lgpio
from time import sleep, perf_counter
import numpy as np

# ====== Konfiguration ======
STEP_PIN   = 18  # BCM
DIR_PIN    = 23
ENABLE_PIN = 24
GPIO_CHIP  = 4   # Haupt-GPIO-Chip auf dem Pi 5

RESET_PIN  = None  # Optional, falls gesteuert
SLEEP_PIN  = None  # Optional, falls gesteuert

STEPS_PER_REV = 200  # Motor 1.8° pro Schritt

# Schritt-Pulseinstellungen (sicher langsam)
PULSE_US     = 10   # Pulsbreite in µs 
STEP_DELAY_S = 0.05    # Abstand zwischen Pulsen (50 ms)
# ==========================

# ====== Funktionen ======
def enable_driver(h, on: bool):
    """ENABLE = LOW -> Treiber aktiv"""
    if on:
        lgpio.gpio_write(h, ENABLE_PIN, 0)
        print("[INFO] Treiber ENABLE aktiviert (LOW)")
    else:
        lgpio.gpio_write(h, ENABLE_PIN, 1)
        print("[INFO] Treiber ENABLE deaktiviert (HIGH)")

def wake_driver(h):
    """RESET/SLEEP auf HIGH setzen (aktiv)"""
    if RESET_PIN is not None:
        lgpio.gpio_write(h, RESET_PIN, 1)
        print("[INFO] RESET HIGH")
    if SLEEP_PIN is not None:
        lgpio.gpio_write(h, SLEEP_PIN, 1)
        print("[INFO] SLEEP HIGH")

def sleep_driver(h):
    if RESET_PIN is not None:
        lgpio.gpio_write(h, RESET_PIN, 0)
        print("[INFO] RESET LOW")
    if SLEEP_PIN is not None:
        lgpio.gpio_write(h, SLEEP_PIN, 0)
        print("[INFO] SLEEP LOW")

def step_once(h, pulse_us=PULSE_US):
    """Ein STEP-Puls, mit Debug"""
    lgpio.gpio_write(h, STEP_PIN, 1)
    # print("[DEBUG] STEP HIGH") # Für sehr langsame Tests ok, sonst zu viel Spam
    sleep(pulse_us / 1_000_000.0)
    lgpio.gpio_write(h, STEP_PIN, 0)
    # print("[DEBUG] STEP LOW")

def step_n(h, n_steps, step_delay_s=STEP_DELAY_S, pulse_us=PULSE_US):
    """N Schritte mit definierter Verzögerung"""
    print(f"[INFO] Fahre {n_steps} Schritte, delay {step_delay_s}s, pulse {pulse_us}us")
    t0 = perf_counter()
    
    # Berechne die reine Wartezeit *zwischen* den Pulsen
    actual_delay_s = step_delay_s - (pulse_us / 1_000_000.0)
    if actual_delay_s < 0:
        actual_delay_s = 0 # Verhindert negative Wartezeit
        print("[WARN] Pulsbreite ist länger als die Schritt-Verzögerung!")

    for i in range(n_steps):
        step_once(h, pulse_us)
        sleep(actual_delay_s)
        
    elapsed = perf_counter() - t0
    print(f"[INFO] Fertig: {n_steps} Schritte in {elapsed:.3f}s")
    return elapsed

def funktion_anfahren():
    print("start")
    t = [1/x for x in range(10, 8000)]
    PULSE_US     = 100 
    for delay in t:
        sleep(delay)
        step_once(h, PULSE_US)
    delay_fix = min(t)
    t0 = perf_counter()
    run_time_s = 1
    while (perf_counter() - t0) < run_time_s:
        sleep(delay_fix)
        step_once(h, PULSE_US)
    for delay in reversed(t):
        sleep(delay)
        step_once(h, PULSE_US)
       


def rotate(h, revolutions=1.0, rpm=30.0):
    """Drehe Motor um Revolutions mit RPM"""
    total_steps = int(round(revolutions * STEPS_PER_REV))
    steps_per_sec = (rpm * STEPS_PER_REV) / 60.0
    step_delay = 1.0 / steps_per_sec
    
    # Pulsbreite muss kürzer sein als die halbe Zykluszeit
    pulse_us = min(PULSE_US, int(step_delay * 1_000_000 * 0.5))
    
    step_n(h, total_steps, step_delay_s=step_delay, pulse_us=pulse_us)

# ====== Hauptprogramm ======
# Wir brauchen einen Handle 'h' für den GPIO-Chip
h = None
try:
    # 1. GPIO-Chip öffnen
    h = lgpio.gpiochip_open(GPIO_CHIP)
    
    # 2. Pins als Ausgang deklarieren
    # lgpio.gpio_claim_output(handle, pin, initial_level)
    lgpio.gpio_claim_output(h, STEP_PIN, 0)
    lgpio.gpio_claim_output(h, DIR_PIN, 0)
    # ENABLE: Start HIGH (disabled) - wir aktivieren ihn manuell
    # Dein gpiozero-Code hat ihn sofort aktiviert (initial_value=True bei active_high=False)
    # Wir machen es hier explizit:
    lgpio.gpio_claim_output(h, ENABLE_PIN, 1) # Starte HIGH (disabled)

    if RESET_PIN is not None:
        lgpio.gpio_claim_output(h, RESET_PIN, 1) # Starte HIGH (wach)
    if SLEEP_PIN is not None:
        lgpio.gpio_claim_output(h, SLEEP_PIN, 1) # Starte HIGH (wach)
        
    print("[INFO] GPIO-Chip geöffnet.")
    
    # --- Beispiel-Ablauf ---
    print("[INFO] Wake Treiber...")
    wake_driver(h)       # Setzt RESET/SLEEP auf HIGH
    enable_driver(h, True) # Setzt ENABLE auf LOW (aktiv)
    
    sleep(0.1) # Kurze Pause, damit der Treiber bereit ist

    # Test 1: Deine manuellen, langsamen Einstellungen (20 Hz)
    print(f"[INFO] Test Rampe")
    lgpio.gpio_write(h, DIR_PIN, 1) # Richtung 0
    funktion_anfahren()

except KeyboardInterrupt:
    print("[WARN] Abbruch per STRG-C")
except Exception as e:
    print(f"[FEHLER] Ein lgpio-Fehler ist aufgetreten: {e}")
    print(f"Stelle sicher, dass 'sudo apt install python3-lgpio' ausgeführt wurde.")
    print(f"Stelle sicher, dass GPIO_CHIP = {GPIO_CHIP} korrekt ist.")

finally:
    # --- Aufräumen ---
    print("[INFO] Treiber deaktivieren und GPIO aufräumen...")
    if h is not None:
        # Treiber sicher deaktivieren
        enable_driver(h, False)
        sleep_driver(h)
        
        # Alle genutzten Pins freigeben
        lgpio.gpio_free(h, STEP_PIN)
        lgpio.gpio_free(h, DIR_PIN)
        lgpio.gpio_free(h, ENABLE_PIN)
        if RESET_PIN is not None:
            lgpio.gpio_free(h, RESET_PIN)
        if SLEEP_PIN is not None:
            lgpio.gpio_free(h, SLEEP_PIN)
            
        # Chip-Verbindung schließen
        lgpio.gpiochip_close(h)
        print("[INFO] GPIO-Chip geschlossen.")