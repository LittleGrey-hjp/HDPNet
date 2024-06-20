# HDPNet
> [!NOTE]  
> The paper is still under review.
> Considering that it may need to be further revised according to the reviewers' comments, the code cannot be stabilized at the moment.
> So it will still take some time for the full code to be released.
## Results
### Quantitative Results
> The final results, which perform very well on the COD10K dataset (contains a lot of small objects and detailed labeling of the objects' fine boundaries)
> 
| Dataset   | $S_m$ | $F_{\omega}$ | $F_m$ | $E_m$ |$E_{x}$|   M   |
| ----------| ----- | -------------| ----- | ----- | ----- | ----- |
| CAMO      | 0.889 | 0.843        | 0.864 | 0.929 | 0.942 | 0.043 |
| CHAMELEON | 0.921 | 0.862        | 0.877 | 0.945 | 0.970 | 0.021 |
| COD10K    | 0.888 | 0.793        | 0.820 | 0.922 | 0.952 | 0.020 |
| NC4K      | 0.901 | 0.848        | 0.870 | 0.932 | 0.950 | 0.029 |
> 
> Other results, which perform very well on the CAMO、CHAMELEON and NC4K dataset, with a slight decline on COD10K.
> 
| Dataset   | $S_m$ | $F_{\omega}$ | $F_m$ | $E_m$ |$E_{x}$|   M   |
| ----------| ----- | -------------| ----- | ----- | ----- | ----- |
| CAMO      | 0.893 | 0.847        | 0.867 | 0.932 | 0.949 | 0.041 |
| CHAMELEON | 0.921 | 0.860        | 0.874 | 0.947 | 0.975 | 0.021 |
| COD10K    | 0.887 | 0.791        | 0.818 | 0.921 | 0.951 | 0.021 |
| NC4K      | 0.903 | 0.848        | 0.870 | 0.932 | 0.952 | 0.029 |

### Qualitative Results
![Qualitative Result](https://github.com/LittleGrey-hjp/HDPNet/blob/main/Visio-camouflage_fig1.jpg)
