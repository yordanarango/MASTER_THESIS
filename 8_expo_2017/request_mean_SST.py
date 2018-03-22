# #!/usr/bin/env python
# from ecmwfapi import ECMWFDataServer
# server = ECMWFDataServer()
# server.retrieve({
#     "class": "ei",
#     "dataset": "interim",
#     "date": "19790101/to/19790301",
#     "expver": "1",
#     "grid": "0.25/0.25",
#     "levtype": "sfc",
#     "param": "34.128",
#     "stream": "moda",
#     "type": "an",
#     "format":"netcdf",
#     "target": "mean_SST.nc",
# })

#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "19790101/to/19790401",
    "expver": "1",
    "grid": "0.25/0.25",
    "area": "90/-180/-90/179.75",
    "levtype": "sfc",
    "param": "34.128",
    "step": "0",
    "stream": "mnth",
    "time": "00:00:00/06:00:00/12:00:00/18:00:00",
    "type": "an",
    "target": "mean_SST.nc",
})

# for i in range(1979, 2017):
#     for j in range(1,13):
#         stR = '%02d' % (j,)

#         print str(i)+stR
