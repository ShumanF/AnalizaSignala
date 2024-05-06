import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import streamlit as st
import librosa
import io
from bokeh.plotting import figure
from bokeh.models import Span

N_uzoraka = 1200000

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

@st.cache_data
def faza_vala_plot(signal):
  signal = signal[:N_uzoraka]
  fig = plt.figure(figsize=(15,8))
  faza = np.diff(signal, prepend=signal[0]) #racunanje razlike hoda x[n+1]−x[n]
  plt.scatter(signal,faza,alpha=0.3)
  plt.title("Fazni Prostor",fontweight="bold")  
  plt.xlabel("Amplituda $(x)$")
  plt.ylabel("Faza ϕ ")
  plt.grid(True)
  plt.show()
  return fig

@st.cache_data
def plot_frekvencijski_spekar(signal,sample_rate):
  signal = signal[:N_uzoraka]
  magnituda = np.abs( np.fft.fft(signal) )
  frequency = np.fft.fftfreq(len(signal), d=1/sample_rate)
  frequency = np.abs(frequency)
  #fig = plt.figure(figsize=(15,7))
  #plt.plot(frequency,magnituda)
  #plt.xlim(0,sample_rate/2)
  #plt.title('Frekvencijiski Spektar',fontsize=19,fontweight='bold')
  #plt.xlabel("Frequency (Hz)",fontsize=14,fontweight='bold')
  #plt.ylabel("Magnituda",fontsize=14,fontweight='bold')
  #plt.grid()
  #plt.show()
  data = pd.DataFrame({'frekvencija':frequency,'magnituda':magnituda})
  return data

@st.cache_data
def gen_plot(signal,Umax,Umin,Udc,Uef,donji_lim,gornji_lim,on):
    fig = plt.figure(figsize=(12,6))
    #plt.style.use('default')  #Solarize_Light2 #default
    fig.patch.set_facecolor('xkcd:cream')
    plt.grid(True)
    plt.xlabel("time[seconds] * sample rate", fontsize=12)
    plt.ylabel("Amplituda $u(t)$ [V]", fontsize=19)
    plt.ylim(donji_lim,gornji_lim)
    if on:
      plt.title("Analiza Generiranog Singalnog Val",fontsize=19,fontweight='bold')
      plt.plot(signal,lw=1.9,color='cornflowerblue')
      plt.axhline(y=Umax,color='r',linestyle=':',label='U max',lw=2)
      plt.axhline(y=Umin,color='darkred',linestyle=':',label='U min',lw=2)
      plt.axhline(y=Udc,color='black',linestyle=':',label='U dc',lw=2)
      plt.axhline(y=Uef,color='b',linestyle='-.',label='U ef',lw=2)
      plt.legend(loc='lower right')
    else:
        plt.title("Generirani Singalni Val",fontsize=19,fontweight='bold')
        plt.plot(signal,lw=2.5,color='royalblue')
    plt.close(fig)
    return fig

@st.cache_resource
def gen_bokeh_plot(t,signal,Umax,Umin,Upp,Udc,Uef,on=False):
  #signal= signal[:10]
  p = figure(title="Generirani Signalni Val",
             x_axis_label=r'\[vrijeme [seconds] \]',
             y_axis_label=r'\[Amplituda  u(t) [V]\]',
             height = 400,width=750,
             y_range=(-Upp,Upp),
             #toolbar_location="above"
             )
  p.border_fill_color = "#ffffc2"
  p.line(t,signal,line_width=2,line_color='#4569d6')
 
  if on:
    p.line(t, Umax, line_color='red', line_dash='dashed', legend_label= f'Umax={Umax:.3f}', line_width=1.5)
    p.line(t, Umin, line_color='red', line_dash='dashed', legend_label= f'Umin={Umin:.3f}', line_width=1.5)
    p.line(t, Udc, line_color='green', line_dash='dashed', legend_label=f'Udc={Udc:.3f}', line_width=1.5)
    p.line(t, Uef, line_color='blue', line_dash='dashed', legend_label=f'Uef={Uef:.3f}', line_width=1.5)
      
  return p
     

@st.cache_data
def generate_sine_wave(amplitude, t, frequency, faza):
    return amplitude * np.sin(2 * np.pi * t * frequency + faza)

@st.cache_data
def generate_cosine_wave(amplitude, t, frequency, faza):
    return amplitude * np.cos(2 * np.pi * t * frequency + faza)

@st.cache_data
def generate_sawtooth_wave(amplitude, t, frequency, faza):
    return amplitude * sp.signal.sawtooth(2 * np.pi * t * frequency + faza, width=1)

@st.cache_data
def generate_triangle_wave(amplitude, t, frequency, faza):
    return amplitude * sp.signal.sawtooth(2 * np.pi * t * frequency + faza, width=0.5)

@st.cache_data
def generate_square_wave(amplitude, t, frequency, faza):
    return amplitude * sp.signal.square(2 * np.pi * t * frequency + faza)

@st.cache_data
def gen_audio(signal,sample_rate):
  # Pretvaranje u 16-bit range
  audio = np.int16(signal * 32767)
  #  Wav file
  buffer = io.BytesIO()
  sp.io.wavfile.write(buffer, sample_rate, audio)
  buffer.seek(0)
  st.audio(buffer,"audio/wav",sample_rate)

def switch_waves(option,amplitude = 1,time = 1,frequency = 1,faza=0,sample_rate = 22050,uploaded_file = None):
  
   t = np.linspace(0, time, int(sample_rate * time), endpoint=False)
   faza = np.deg2rad(faza)
   options = {
            "Sin": generate_sine_wave(amplitude, t, frequency, faza),
            "Cos": generate_cosine_wave(amplitude, t, frequency, faza),
            "Sawtooth": generate_sawtooth_wave(amplitude, t, frequency, faza),
            "Triangle": generate_triangle_wave(amplitude, t, frequency, faza),
            "Square": generate_square_wave(amplitude, t, frequency, faza),
            "White noise":  white_noise(amplitude,time,sample_rate),
            "Brown noise":  brown_noise(time,sample_rate),
            "Uploaded File": None if uploaded_file is None else librosa.load(uploaded_file,sr=sample_rate)[0]

   }
   
   return options.get(option,"Default")



if __name__ == '__main__':


  st.header("ANALIZA SIGNALA")

  
  gen_y1 = st.sidebar.button("Generate y1")
  gen_y2 = st.sidebar.button("Generate y2")
  
  sample_rate = st.sidebar.slider("Odaberi SAMPLE RATE [Hz]", 1, 96000,22050) #Default SR 22050Hz, Max 96k Hz   
  st.sidebar.info("Frekvencija se resetira promjenom sampling-a (Nyquist frequiency)")  
  
  amplitude = st.sidebar.number_input("Odaberi amplitudu [V]", 0.1, 1000.0,1.)
  frequency = st.sidebar.number_input("Odaberi frekevenciju [Hz]", 1, int(sample_rate/2)) #nyquist frequency
  time = st.sidebar.number_input("Odaberi trajanje u sekundama [s]", 1, 60)
  faza = st.sidebar.slider("Odaberi fazu signala (stupnjevi)",-360,360,0)
  
  prigušenje = st.sidebar.slider("Prigusenje vala formulom $$e^{-\lambda t} \cdot signal$$ slider podesava λ :",0.0, 10.0, 0.0)
  
  pick_wave_gen = st.sidebar.radio(
    "Odaberi val koji generirati",
    [
        "Sin","Cos","Sawtooth","Triangle","Square","White noise","Brown noise","Uploaded File"
    ]
    )
  
  uploaded_file = st.sidebar.file_uploader("Odaberi audio file [wav,mp3]", type=['wav', 'mp3'])
    
  signal = switch_waves(pick_wave_gen,amplitude,time,frequency,faza,sample_rate,uploaded_file=uploaded_file)

  #generiranje = st.sidebar.toggle("Lijevo y1 Desno y2")
  
  t = np.linspace(0, time, (sample_rate * time)) 

  if pick_wave_gen == 'Uploaded File':
     t = len(signal)
  
  signal = np.exp(-t*prigušenje) * signal

  Umax = max(signal)
  Umin = min(signal)
  Upp = Umax-Umin
  Udc = dc_component(signal)
  Uef = efective(signal)
  standard_deviation = sp.ndimage.standard_deviation(signal[:N_uzoraka])
  #gamma = 'inf' if (standard_deviation/Udc) > 100000.0 else (standard_deviation/Udc) * 100
  gamma = float('inf') if (standard_deviation/Udc) > 100000.0 else (standard_deviation/Udc) * 100
  Psr = Uef**2
  Psr_dBW = 20 * np.log10(Uef)

  start,end = st.slider('Podesi slider za vremenski zoom na val (skalirano je po sampling * vrijeme, slider ide od 0 do t * N samples)',0, sample_rate*time,(0,sample_rate))    
  #donji_lim,gornji_lim = st.slider('Podesi slider za ublizavanje na amplitude',-Upp,Upp,(-Upp,Upp))

  dugme  = st.toggle('Prikazi na grafu Upp,Udc,Uef')
  
  #col1, col2, col3 = st.columns(3)
  #col1.button("y1 + y2")
  #col2.button("y1 * y2")
  #col3.button("Undo")
  
  
  #st.write(gen_plot(signal[start:end],Umax,Umin,Udc,Uef,donji_lim=donji_lim,gornji_lim=gornji_lim,on=dugme))

  if 'y1' not in st.session_state:
    st.session_state.y1 = np.sin(2*np.pi*t) #primjerni val kada se ucitava stranica
  if 'y2' not in st.session_state:
    st.session_state.y2 = []              #drugi val je prazan dok ga korisnik ne definira

  if gen_y1:
    st.session_state.y1 = signal
  if gen_y2:
     st.session_state.y2 = signal
  
  t = t[start:end]; y1 = st.session_state.y1[start:end]; y2 = st.session_state.y2[start:end]

  st.bokeh_chart(gen_bokeh_plot(t,y1,Umax,Umin,Upp,Udc,Uef,on=dugme),use_container_width = False) # prvi osnovni val
  st.subheader('Zvuk #1 generiranog signala')
  gen_audio(y1,sample_rate)

  if len(y2) != 0: 
    st.bokeh_chart(gen_bokeh_plot(t,y2,Umax,Umin,Upp,Udc,Uef,on=False),use_container_width=False) # drugi val po izboru
    brisanje = st.button('Izbrisi graf #2 (stisni dva puta)')
    if brisanje:
      st.session_state.y2 = []
    st.subheader('Zvuk #2 generiranog signala')
    gen_audio(y2,sample_rate)
    

  st.write("Analiza signala [uzima se do prvih 1,2M uzoraka]")
        
  rounded_values = [str(round(x,7)) for x in [Umax,Umin,Upp,Udc,Uef,standard_deviation,gamma,Psr,Psr_dBW ]]
          

  #st.write(gen_plot(signal[start:end],Umax,Umin,Udc,Uef,donji_lim=donji_lim,gornji_lim=gornji_lim,on=True))
  
  dataframe = pd.DataFrame(
    {
    "Formule": [
        "https://raw.githubusercontent.com/ShumanF/AnalizaSignala/master/Formule/1.png",
        "https://raw.githubusercontent.com/ShumanF/AnalizaSignala/master/Formule/2.png",
        "https://raw.githubusercontent.com/ShumanF/AnalizaSignala/master/Formule/3.png",
        "https://raw.githubusercontent.com/ShumanF/AnalizaSignala/master/Formule/4.png",
        "https://raw.githubusercontent.com/ShumanF/AnalizaSignala/master/Formule/5.png",
        "https://raw.githubusercontent.com/ShumanF/AnalizaSignala/master/Formule/6.png",
        "https://raw.githubusercontent.com/ShumanF/AnalizaSignala/master/Formule/7.png",
        "https://raw.githubusercontent.com/ShumanF/AnalizaSignala/master/Formule/8.png",
        "https://raw.githubusercontent.com/ShumanF/AnalizaSignala/master/Formule/9.png",
      ],
  "Values":["Umax [V]","Umin [V]","Upp [V]",
  "Udc [V]","Uef [V]","Stanardna devijacija: σ [V]",
  "Faktor valovitosti: γ [%]","Srednja snaga na 1 Ω: Psr/SNR [W]","Psr [dBW] "],
  "":(rounded_values),
      
    }
)
  st.dataframe(dataframe,
               use_container_width=True,
               hide_index=True,
               column_config={"Formule":st.column_config.ImageColumn("Formule")},
              
              )
  faza_on = st.checkbox("Fazna karakteristike singala [uzima se prvih 1,2M uzoraka signala] (stisni gumb)")
  frequency_spectrum = st.checkbox("Frekvencijski spektar singala [uzima se prvih 1,2M uzoraka signala] (stisni gumb)")
  if faza_on:
    st.write(faza_vala_plot(signal))
  if frequency_spectrum:
     st.line_chart((plot_frekvencijski_spekar(signal,sample_rate)), x='frekvencija', y = 'magnituda')

