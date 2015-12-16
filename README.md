## About

Tool to test the ```carmcmc``` package by B. C. Kelly on multiband long-term lightcurves of 3C273 from
the Soldi et al. (2008) catalog: http://isdc.unige.ch/3c273/

Results presented in 5th Fermi Symposium:
http://fermi.gsfc.nasa.gov/science/mtgs/symposia/2014/program/14B_Sobolewska.pdf

3C273 lightcurves and carma_sample pickle files stored in ```data/``` directory (not included in this repository).

## Usage

* To generate the fractional variability spectrum (Slide 13 of the Fermi Symposium presentation):

```
import var
var.plot_fvar()
```
