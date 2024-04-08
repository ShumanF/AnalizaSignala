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

def white_noise(amplitude,time, sample):
    noise = 2*amplitude * np.random.random(int(time * sample)) - (amplitude)
    return noise

def brown_noise(time, sample):
    noise = np.random.normal(0,1,int(time*sample))
    noise = np.cumsum(noise)
    noise = noise - np.mean(noise)  # Remove DC offset
    noise = noise / np.max(np.abs(noise)) #normalisation
    return noise

def dc_component(y):
  integral = np.trapz(y)
  return abs(integral / len(y))

def efective(y):
  integral = np.trapz(y**2)
  return np.sqrt(integral / len(y))

def gen_plot(y,Umax,Umin,Udc,Uef,on):
    fig = plt.figure(figsize=(12,6))
    plt.style.use('Solarize_Light2')  #Solarize_Light2 #default
    #plt.grid()
    if on:
      plt.plot(y,lw=1,color='cornflowerblue')
      plt.axhline(y=Umax,color='r',linestyle=':',label='U max',lw=2)
      plt.axhline(y=Umin,color='darkred',linestyle=':',label='U min',lw=2)
      plt.axhline(y=Udc,color='black',linestyle=':',label='U dc',lw=2)
      plt.axhline(y=Uef,color='b',linestyle='-.',label='U ef',lw=2)
      plt.legend(loc='lower right')
    else:
        plt.plot(y,lw=2.5,color='royalblue')
    plt.close(fig)
    return fig

#def generate_wave(y):
  
    
def switch_waves(option,amplitude = 1,time = 1,frequency = 1,sample_rate = 44100 ):
   
   t = np.linspace(0, time, int(sample_rate * time), endpoint=False)
   options = {
            "Sin": amplitude* np.sin(2*np.pi*t*frequency),
            "Cos":  amplitude*np.cos(2*np.pi*t*frequency),
            "Sawtooth": amplitude*sp.signal.sawtooth(2 * np.pi *t*frequency,1),
            "Tirangle": amplitude*sp.signal.sawtooth(2 * np.pi * t*frequency,0.5),
            "Square": amplitude*sp.signal.square(2*np.pi*t*frequency),
            "White noise":  white_noise(amplitude,time,sample_rate),
            "Brown noise":  brown_noise(time,sample_rate),
            #"Uploaded File": audio_file

   }
   return options.get(option,"Default")

def tabela_analize(y):
   y = y[:N_uzoraka] #analiza prvih N definiranih uzoraka

   Umax = (max(y))
   Umin = (min(y))
   Upp = (Umax - Umin)
   Udc = (dc_component(y))
   Uef = (efective(y))
   standard_deviation = (sp.ndimage.standard_deviation(y))
   gamma = float('inf') if (standard_deviation/Udc) > 100000.0 else (standard_deviation/Udc) * 100
   Psr = (Uef**2) 
   Psr_dBW = (20 * np.log10(Uef)) 
   rounded_values = [Umax,Umin,Upp,Udc,Uef,standard_deviation,gamma,Psr,Psr_dBW ] 
   table = pd.DataFrame( {
                              "Values":rounded_values,
                              #"":["V","V","V","V","V","%"]
                              },
                            index= ["Umax","Umin","Upp","Udc","Uef","σ","γ","Psr","Psr_dBW "])

   return table


if __name__ == '__main__':

  st.header("ANALIZA SIGNALA")

  sample_rate = st.sidebar.slider("Odaberi SAMPLE RATE [Hz]", 1, 44100,44100)    
  amplitude = st.sidebar.number_input("Odaberi amplitudu [V]", 0.1, 230.0,1.)
  st.sidebar.info("Frekvencija se resetira promjenom sampling-a (Nyquist frequiency)")
  frequency = st.sidebar.number_input("Odaberi frekevenciju [Hz]", 1, int(sample_rate/2)) #nyquist frequency
  time = st.sidebar.number_input("Odaberi trajanje u sekundama [s]", 1, 60)

  pick_wave_gen = st.sidebar.radio(
    "Odaberi val koji generirati",
    [
        "Sin","Cos","Sawtooth","Triangle [KVAR!]","Square","White noise","Brown noise","Uploaded File [KVAR!]"
    ]
    )
  uploaded_file = st.sidebar.file_uploader("Odaberi audio file [wav,mp3]", type=['wav', 'mp3'])

  if uploaded_file is not None:
    audio_file = librosa.load(uploaded_file,sr=sample_rate)
    
  y = switch_waves(pick_wave_gen,amplitude,time,frequency,sample_rate)
  
  start,end = st.slider('Podesi slider za ublizavanje na val (skalirano je po sampling * vrijeme, slider ide od 0 do N samples)',0, sample_rate*time,(0,sample_rate))    

  st.write(gen_plot(y[start:end],Umax=0,Umin=0,Udc=0,Uef=0,on=False))

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
    #rounded_values =  [Umax,Umin,Upp,Udc,Uef,standard_deviation,gamma,Psr,Psr_dBW ]
    
    index= ["Umax","Umin","Upp","Udc","Uef","σ[stanardna devijacija]","γ [faktor valovitosti]","Psr/SNR","Psr_dBW "]
    mjerne_jedinice = ['v',"V","V","V","V","V","%","W","dBW"]
    data = []
    for i in range (1,9):
      data.append({
        #"Formule":f"Formule/{i}.png",
        "Values":index[i],
        "":rounded_values[i],
        "mjerne_jedinice":mjerne_jedinice[i]
      })
    ttable = pd.DataFrame(data)
    
    table = pd.DataFrame( {
                                  "Values":rounded_values,
                                  #"":["V","V","V","V","V","%"]
                                  },
                                index= ["Umax","Umin","Upp","Udc","Uef","σ","γ","Psr","Psr_dBW "])

    st.write(gen_plot(y[start:end],Umax,Umin,Udc,Uef,checkbox))
    #st.dataframe(ttable,width=300,hide_index=True)

    st.data_editor(ttable,
                   num_rows='fixed',
                   hide_index=1,
                   column_config={"Forumule":st.column_config.ImageColumn("Formule")},
                   width=400
                  )
    #st.markdown(ttable.to_html(escape=False), unsafe_allow_html=True)

   

