Tool to test the CARMA package on multiband long-term lightcurves of 3C273 from
the Soldi et al. (2008) catalog: http://isdc.unige.ch/3c273/

Results presented in 5th Fermi Symposium:
http://fermi.gsfc.nasa.gov/science/mtgs/symposia/2014/program/14B_Sobolewska.pdf

Directory data/ contains ascii files with 3C273 lightcurves and carma_sample pickle files.

* To generate the fractional variability spectrum plot on Slide 13 of the Fermi Symposium
presentation:

```
import var
var.plot_fvar()
```
