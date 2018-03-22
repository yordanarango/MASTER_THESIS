#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "1979-01-01/to/2017-12-31",
    "expver": "1",
    "grid": "0.25/0.25",
    "levtype": "sfc",
    "param": "34.128",
    "step": "0",
    "stream": "oper",
    "time": "00:00:00/06:00:00/12:00:00/18:00:00",
    "type": "an",
    'area': "18/-100/0/-75",
    "format" : "netcdf",
    "target": "/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/SST_EOFs.nc",
})
