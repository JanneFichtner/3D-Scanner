import smbus2
import time

I2C_BUS = 3
I2C_ADDR = 0x08

# Befehls-Definitionen (müssen mit Arduino übereinstimmen)
CMD_BLINK = 1
CMD_MOVE  = 2

def main():
    try:
        bus = smbus2.SMBus(I2C_BUS)
        
        # --- BEISPIEL: "MOVE, 2000" senden ---
        
        ziel_wert = 200 * 16 * 5 * 4
        
        # Wir zerlegen 2000 in zwei Bytes
        # 2000 in Binär ist: 00000111 11010000
        # High Byte: 00000111 (7)
        # Low Byte:  11010000 (208)
        
        high_byte = (ziel_wert >> 8) & 0xFF
        low_byte = ziel_wert & 0xFF
        
        print(f"Sende MOVE Befehl mit Wert {ziel_wert}...")
        
        # write_i2c_block_data(Adresse, Register/Befehl, [DatenListe])
        # Wir nutzen den "Befehl" als Register-Adresse und die Bytes als Daten
        bus.write_i2c_block_data(I2C_ADDR, CMD_MOVE, [high_byte, low_byte])
        
        print("Gesendet.")
        bus.close()
        
    except Exception as e:
        print(f"Fehler: {e}")

if __name__ == "__main__":
    main()