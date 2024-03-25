import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import streamlit as st


# Parameters
sample_rate = 44100  # Sample rate in Hz
duration = 6.28  # Duration in seconds
frequency = 440  # Frequency of the wave in Hz
num_samples = sample_rate * duration

def white_noise(time, sample):
    noise = np.random.random(int(time * sample))
    return noise

def brown_noise(time, sample):
    noise = np.random.normal(0,1,int(sample*time))
    noise = np.cumsum(noise)
    noise = noise - np.mean(noise)  # Remove DC offset
    noise = noise / np.max(np.abs(noise)) #normalisation
    return noise

def dc_component(y):
  integral = np.trapz(y)
  return integral / len(y)

def efective(y):
  integral = np.trapz(y**2)
  return np.sqrt(integral / len(y))


t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

white = white_noise(2,44100)
brown = brown_noise(2,44100)
sin = np.sin(2*np.pi*t)
cos = np.cos(2*np.pi*t)
square = sp.signal.square(2*np.pi*t)
sawtooth = sp.signal.sawtooth(2 * np.pi *t,1)
triangle = sp.signal.sawtooth(2 * np.pi * t,0.5)

def generate_tone():
  # Ensure the values are in 16-bit range
  audio = np.int16(y * 32767)
  # Write to a MP3 file
  sp.io.wavfile.write('wave.mp3', sample_rate, audio)

def switch_waves(option):
   options = {
            "Sin": np.sin(2*np.pi*t),
            "Cos":np.cos(2*np.pi*t),
            "Sawtooth":sp.signal.sawtooth(2 * np.pi *t,1),
            "Tirangle":sp.signal.sawtooth(2 * np.pi * t,0.5),
            "Square":sp.signal.square(2*np.pi*t),
            "White noise":white_noise(2,44100)[:500],
            "Brown noise":brown_noise(2,44100)
   }
   return options.get(option,"Default")

pick_wave_gen = st.sidebar.radio(
   "Odaberi val koji generirati",
   [
      "Sin","Cos","Sawtooth","Tirangle","Square","White noise","Brown noise"
   ]
)
on = st.sidebar.checkbox("Napravi analizu signala") 
#st.sidebar.toggle("Napravi analizu signala")


y = switch_waves(pick_wave_gen)
Umax = max(y)
Umin = min(y)
Udc = dc_component(y)
Uef = efective(y)
σ = sp.ndimage.standard_deviation(y)

table =pd.DataFrame( {
                      "Values":[Umax,Umin,Udc,Uef,σ],
                      "":["V","V","V","V","V"]
                      },
                    index= ["Umax","Umin","Udc","Uef","σ"])

fig = plt.figure(figsize=(12,6))
plt.style.use('Solarize_Light2') 
#plt.grid()
if on:
  plt.plot(y,lw=1,color='cornflowerblue')
  plt.axhline(y=Umax,color='r',linestyle=':',label='U max',lw=2)
  plt.axhline(y=Umin,color='darkred',linestyle=':',label='U min',lw=2)
  plt.axhline(y=Udc,color='black',linestyle=':',label='U dc',lw=2)
  plt.axhline(y=Uef,color='b',linestyle='-.',label='U ef',lw=2)
  plt.legend(loc='lower right')
else:
     plt.plot(y,lw=1.5,color='blue')
img = fig.savefig('plot.png')
plt.close(fig)

#col1, col2 = st.columns(2)

st.header("Analiza Signala")

st.sidebar.caption("***Gangsta***")
st.sidebar.dataframe(table)

st.image('plot.png')
st.write(fig)

#st.slider('Select a range of values',0, num_samples, (5,50))

st.button("Generate Tone",on_click=generate_tone())
st.audio('wave.mp3')