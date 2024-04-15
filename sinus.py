import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import streamlit as st
import io
import os
import librosa

N_uzoraka = 1200000
wave_list = []

@st.cache_data
def white_noise(amplitude,time, sample):
    noise = 2*amplitude * np.random.random(int(time * sample)) - (amplitude)
    return noise.astype(np.float16)

@st.cache_data
def brown_noise(time, sample):
    noise = np.cumsum(np.random.normal(0,1,int(time*sample)))
    noise = noise - np.mean(noise)  # Remove DC offset
    noise = noise / np.max(np.abs(noise)) #normalisation
    return noise.astype(np.float16)

def dc_component(y):
  integral = np.trapz(y)
  return abs(integral / len(y))

def efective(y):
  integral = np.trapz(y**2)
  return np.sqrt(integral / len(y))

def gen_plot(y,Umax,Umin,Udc,Uef,on):
    fig = plt.figure(figsize=(12,6))
    #plt.style.use('default')  #Solarize_Light2 #default
    fig.patch.set_facecolor('xkcd:cream')
    plt.grid()
    plt.xlabel("time[seconds] * sample rate", fontsize=12,fontname='Arial')
    plt.ylabel("Amplituda $u(t)$ [V]", fontsize=19,fontname='Arial')
    if on:
      plt.title("Analiza Generiranog Singalnog Val",fontsize=19,fontweight='bold')
      plt.plot(y,lw=1.9,color='cornflowerblue')
      plt.axhline(y=Umax,color='r',linestyle=':',label='U max',lw=2)
      plt.axhline(y=Umin,color='darkred',linestyle=':',label='U min',lw=2)
      plt.axhline(y=Udc,color='black',linestyle=':',label='U dc',lw=2)
      plt.axhline(y=Uef,color='b',linestyle='-.',label='U ef',lw=2)
      plt.legend(loc='lower right')
    else:
        plt.title("Generirani Singalni Val",fontsize=19,fontweight='bold')
        plt.plot(y,lw=2.5,color='royalblue')
    plt.close(fig)
    return fig

#def generate_wave(y):

@st.cache_data
def generate_sine_wave(amplitude, frequency, t):
    return amplitude * np.sin(2 * np.pi * t * frequency)
@st.cache_data
def generate_cosine_wave(amplitude, frequency, t):
    return amplitude * np.cos(2 * np.pi * t * frequency)
@st.cache_data
def generate_sawtooth_wave(amplitude, frequency, t):
    return amplitude * sp.signal.sawtooth(2 * np.pi * t * frequency, width=1)
@st.cache_data
def generate_triangle_wave(amplitude, frequency, t):
    return amplitude * sp.signal.sawtooth(2 * np.pi * t * frequency, width=0.5)
@st.cache_data
def generate_square_wave(amplitude, frequency, t):
    return amplitude * sp.signal.square(2 * np.pi * t * frequency)

    
def switch_waves(option,amplitude = 1,time = 1,frequency = 1,sample_rate = 44100,uploaded_file = None ):
  
   t = np.linspace(0, time, int(sample_rate * time), endpoint=False)
   options = {
            "Sin": generate_sine_wave(amplitude,frequency,t),
            "Cos":  generate_cosine_wave(amplitude,frequency,t),
            "Sawtooth": generate_sawtooth_wave(amplitude,frequency,t),
            "Triangle": generate_triangle_wave(amplitude, frequency, t),
            "Square": generate_square_wave(amplitude, frequency, t),
            "White noise":  white_noise(amplitude,time,sample_rate),
            "Brown noise":  brown_noise(time,sample_rate),
            #"Uploaded File": librosa.load('C:/Users/Korisnik/Desktop/AnalizaSignala-1/wave.mp3')

   }
   return options.get(option,"Default")


if __name__ == '__main__':

  st.header("ANALIZA SIGNALA")

  sample_rate = st.sidebar.slider("Odaberi SAMPLE RATE [Hz]", 1, 44100,44100)    
  amplitude = st.sidebar.number_input("Odaberi amplitudu [V]", 0.1, 1000.0,1.)
  st.sidebar.info("Frekvencija se resetira promjenom sampling-a (Nyquist frequiency)")
  frequency = st.sidebar.number_input("Odaberi frekevenciju [Hz]", 1, int(sample_rate/2)) #nyquist frequency
  time = st.sidebar.number_input("Odaberi trajanje u sekundama [s]", 1, 60)

  pick_wave_gen = st.sidebar.radio(
    "Odaberi val koji generirati",
    [
        "Sin","Cos","Sawtooth","Triangle","Square","White noise","Brown noise","Uploaded File"
    ]
    )
  uploaded_file = st.sidebar.file_uploader("Odaberi audio file [wav,mp3]", type=['wav', 'mp3'])

  if uploaded_file is not None:
    audio_file = librosa.load(uploaded_file,sr=sample_rate)
    
  y = switch_waves(pick_wave_gen,amplitude,time,frequency,sample_rate,uploaded_file='wave.mp3')
  
  start,end = st.slider('Podesi slider za ublizavanje na val (skalirano je po sampling * vrijeme, slider ide od 0 do N samples)',0, sample_rate*time,(0,sample_rate))    

  st.write(gen_plot(y[start:end],Umax=0,Umin=0,Udc=0,Uef=0,on=False))
  #st.write(type(y))
  # Ensure the values are in 16-bit range
  audio = np.int16(y * 32767)
  # Write to a MP3 file
  sp.io.wavfile.write('wave.mp3', sample_rate, audio)
  st.subheader('Zvuk generiranog signala')
  st.audio('wave.mp3')

  checkbox = st.toggle("Stisnite gumb da napravite analizu signala [uzima se do prvih 1,2M uzoraka]",value=True)

  if checkbox:
    Umax = max(y[:N_uzoraka])
    Umin = min(y[:N_uzoraka])
    Upp = Umax - Umin
    Udc = dc_component(y)
    Uef = efective(y)
    standard_deviation = sp.ndimage.standard_deviation(y[:N_uzoraka])
    #gamma = 'inf' if (standard_deviation/Udc) > 100000.0 else (standard_deviation/Udc) * 100
    gamma = float('inf') if (standard_deviation/Udc) > 100000.0 else (standard_deviation/Udc) * 100
    Psr = Uef**2
    Psr_dBW = 20 * np.log10(Uef) 
  
    rounded_values = [round(x,16) for x in [Umax,Umin,Upp,Udc,Uef,standard_deviation,gamma,Psr,Psr_dBW ]]
    
    index= ["Umax","Umin","Upp","Udc","Uef","σ[stanardna devijacija]","γ [faktor valovitosti]","Psr/SNR","Psr_dBW "]
    mjerne_jedinice = ["V","V","V","V","V","V","%","W","dBW"]
    #forumule = [st.latex(r'''U_{ef} = U_{RMS} = \sqrt{\frac{1}{T} \int_{T} u^2(t) dt}''')]

    data = []
    for i in range (0,9):
      data.append({
        #"Formule":rf"C:\Users\Korisnik\Desktop\AnalizaSignala-1\Formule/{i}.png",
        "Values":index[i],
        "":rounded_values[i],
        "mjerne_jedinice":mjerne_jedinice[i]
      })
    table = pd.DataFrame(data)
    
  
    st.write(gen_plot(y[start:end],Umax,Umin,Udc,Uef,checkbox))
    #st.dataframe(ttable,width=300,hide_index=True)

    st.dataframe(table,
                 hide_index=1,
                 column_config={"Forumule":st.column_config.ImageColumn("Formule")},
                 width=300
                )