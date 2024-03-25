import matplotlib.pyplot as plt
import io

def plot_img(y,Umax,Umin,Udc,Uef,on):
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
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close(fig)
    img_buffer.seek(0)
    
    return img_buffer