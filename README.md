# HDPNet
> [!NOTE]  
> The paper is still under review.
> Considering that it may need to be further revised according to the reviewers' comments, the code cannot be stabilized at the moment.
> So it will still take some time for the full code to be released.
## Results
### Quantitative Results
> 
> Our final results,  which perform very well on the COD10K dataset (contains a lot of small objects and detailed labeling of the objects' fine boundaries).
>
> we adopt five kinds of evaluation metrics:
> 
> S-measure($S_m$), weighted F-measure($F_{\omega}$), adaptive F-measure($F^a_m$), mean F-measure($F^m_m$),max F-measure($F^x_m$), adaptive E-measure($E^a_m$),
> mean E-measure($E^m_m$), max E-measure ($E^x_m$), and mean absolute error($\mathcal{M}$)
> 
| Dataset   | $S_m \uparrow$ | $F_{\omega} \uparrow$ | $F^a_m \uparrow$ | $F^m_m \uparrow$ | $F^x_m \uparrow$ | $E^a_m \uparrow$ | $E^m_m \uparrow$ | $E^x_m \uparrow$ | $\mathcal{M} \downarrow$ |
|:---------:|:--------------:|:--------------------:|:----------------:|:----------------:|:----------------:|:----------------:|:----------------:|:----------------:|:-------------------------:|
| CAMO      |    0.893       |        0.851         |      0.848       |      0.870       |      0.890       |      0.932       |      0.934       |      0.948       |           0.040           |
| CHAMELEON |    0.921       |        0.861         |      0.849       |      0.874       |      0.902       |      0.943       |      0.947       |      0.970       |           0.021           |
| COD10K    |    0.888       |        0.794         |      0.770       |      0.820       |      0.852       |      0.915       |      0.925       |      0.951       |           0.020           |
| NC4K      |    0.902       |        0.850         |      0.845       |      0.871       |      0.891       |      0.931       |      0.934       |      0.950       |           0.029           |
> 
### Qualitative Results
![Qualitative Result](https://github.com/LittleGrey-hjp/HDPNet/blob/main/Visio-camouflage_fig1.jpg)
