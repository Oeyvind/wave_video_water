from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 8000)

def send_wave_data(freqs, direction):
    client.send_message("/wave/freq_low", float(freqs["low"]))
    client.send_message("/wave/freq_mid", float(freqs["mid"]))
    client.send_message("/wave/freq_high", float(freqs["high"]))
    client.send_message("/wave/direction", float(direction))