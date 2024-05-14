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
  signal = signal[:N_uzoraka:5]
  faza = np.diff(signal, prepend=signal[0]) #racunanje razlike hoda x[n+1]−x[n]
  return pd.DataFrame({'Signal':signal,'Faza':faza})

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

def plot_histogram_amplitude(signal,N_bins,Amin, Amax):

  # Srijedna vrijednost i stand devijacija signala
  mu, sigma = sp.stats.norm.fit(signal)

  # Generate normal PDF/CDF
  y_norm = np.linspace(Amin, Amax,1000)
  pdf_norm = sp.stats.norm.pdf(y_norm, mu, sigma)

  # Plot histogram & normal distribution
  fig, ax = plt.subplots(figsize=(10, 5))
  counts, edges, patches = ax.hist(signal, bins=N_bins, range=(Amin, Amax),density=True,histtype='barstacked', alpha=0.6,rwidth=0.84,edgecolor='blue', color='royalblue', label='Histogram of $y = \sin(x)$')
  ax.plot(y_norm, pdf_norm, 'r-', linewidth=2, label=f'Fitted Normal Dist. ($\mu={mu:.2f}$, $\sigma={sigma:.2f}$)')

  # Linija za srednju vrijednost
  ax.axvline(mu, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mu:.2f}')

  # Linije za stand devijaciju
  ax.axvline(mu - sigma, color='green', linestyle='dashed', linewidth=2, label=f'Std Dev: {sigma:.2f}')
  ax.axvline(mu + sigma, color='green', linestyle='dashed', linewidth=2)

  # Label each bin
  bin_labels = [f"[{edges[i]:.1f}, {edges[i+1]:.1f}>" for i in range(len(edges)-1)]
  ax.set_xticks(edges[:-1] + np.diff(edges)/2)  # Oznaci sve na sredinu binova
  ax.set_xticklabels(bin_labels, rotation=90)  # Rotiranje horizontalno da se vidi sve

  plt.xlim(Amin, Amax)
  plt.legend()
  plt.grid(True)
  #plt.show()
  return st.write(fig)


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
def gen_bokeh_plot(t,signal,Udc,Uef,on=False):
  #signal= signal[:10]
  Umax=max(signal); Umin=min(signal)
  Upp = max(signal) - min(signal)
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

  start,end = st.slider('Podesi slider za vremenski zoom na val (skalirano je po sampling * vrijeme, slider ide od 0 do t * N samples)'
                        ,0, sample_rate*time,
                        (0,sample_rate)
                        )    
  #donji_lim,gornji_lim = st.slider('Podesi slider za ublizavanje na amplitude',-Upp,Upp,(-Upp,Upp))

  dugme  = st.toggle('Prikazi na grafu Upp,Udc,Uef')
  
  #col1, col2, col3 = st.columns(3)
  #zbrajanje = col1.button("y1 + y2")
  #mnozenje = col2.button("y1 * y2")
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

  tab1, tab2, tab3, tab4 = st.tabs(["Main", "Zbrajanje", "Mnozenje","Lissajousove krivulje "])

  with tab1:
    # prvi osnovni val
    st.bokeh_chart(gen_bokeh_plot(t,y1,Udc,Uef,on=dugme),use_container_width = False)
    st.subheader('Zvuk #1 generiranog signala')
    gen_audio(y1,44100)

    # drugi val po izboru
    if len(y2) != 0: 
      st.bokeh_chart(gen_bokeh_plot(t,y2,Udc,Uef,on=False),use_container_width=False) 
      brisanje = st.button('Izbrisi graf #2 (stisni dva puta)')
      if brisanje:
        st.session_state.y2 = []
      st.subheader('Zvuk #2 generiranog signala')
      gen_audio(y2,44100)

  with tab2:
    if len(y2) != 0:
      sum = y1 + y2
      st.bokeh_chart(gen_bokeh_plot(t,sum,Udc=0,Uef=0),use_container_width=True)
      #gen_audio(sum,44100)  
    else:
      st.write("GENERIRAJ DRUGI SIGNAL ZA REZULTATE")
      st.markdown("""---""")


  with tab3:
    if len(y2) != 0:
      mul = y1 * y2
      st.bokeh_chart(gen_bokeh_plot(t,mul,Udc=0,Uef=0),use_container_width=True)
      #gen_audio(sum,44100)
    else:
        st.write("GENERIRAJ DRUGI SIGNAL ZA REZULTATE")
        st.markdown("""---""") 
 

  with tab4:
     if len(y2) != 0:
      st.scatter_chart(pd.DataFrame({'y1':y1,'y2':y2})[::10],
                        x='y1',
                        y='y2',
                        color='#1cc91c',
                        size=25
                        #height=400,
                        #width=700
                        #use_container_width=True
                        ) 
     else:
        st.write("GENERIRAJ DRUGI SIGNAL ZA REZULTATE")
        st.markdown("""---""") 

  st.markdown("""---""")
  st.write("Analiza #1 signala [uzima se do prvih 1,2M uzoraka]")
        
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
  st.markdown("""---""")
  faza_on = st.checkbox("Fazna karakteristike singala [uzima se prvih 1,2M uzoraka signala] (stisni gumb)")
  frequency_spectrum = st.checkbox("Frekvencijski spektar singala [uzima se prvih 1,2M uzoraka signala] (stisni gumb)")
  histogram_amplitude = st.checkbox("Amplitudni histogram")

  if faza_on:
    st.write("FAZNI PROSTOR")
    st.scatter_chart(faza_vala_plot(y1),x='Signal',y='Faza',size=25)

  st.markdown("""---""")

  if frequency_spectrum:
     st.write("FREKVENCIJSKI SPEKTAR")
     st.line_chart((plot_frekvencijski_spekar(y1,sample_rate)), x='frekvencija', y = 'magnituda')

  if histogram_amplitude:
    st.write("Histogram Amplitude")
    signal = y1
    if(len(y2)!=0):
      odabran_signal = st.radio("Odaberi da analiziras prvi ili drugi singal", ["y1","y2"])
      if odabran_signal == 'y2':
         signal = y2      

    col1, col2 = st.columns(2)
    
    with col1:
     N_bins = st.number_input("Odaberi broj razreda",3,99,11) # od 3 do 99, Default 11 bins

    with col2:
      Amin, Amax = st.slider("Podesi slider za ulaze Amax i Amin",min(signal),max(signal),(min(signal),max(signal)))
    plot_histogram_amplitude(signal,N_bins,Amin, Amax)
