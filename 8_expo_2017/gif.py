import subprocess
from os import listdir

#comand1 = 'convert -delay 20 -loop 0  /home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/8_expo_2017/COMPOSITES_GEO_POT/200hPa/PN/Abr/PN_MEAN_25_50_DUR_50_75/*.png /home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/8_expo_2017/COMPOSITES_GEO_POT/200hPa/PN/Abr/PN_MEAN_25_50_DUR_50_75/myimage.gif'

#command2 = 'cd /Escritorio'
#command2 = 'ls /home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/8_expo_2017/COMPOSITES_GEO_POT/200hPa/PN/Abr'

#process1 = subprocess.Popen(command1.split(), stdout=subprocess.PIPE)
#process2 = subprocess.Popen(command2.split(), stdout=subprocess.PIPE)

#output, error = process2.communicate()
#print output
#print error


path = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/8_expo_2017/COMPOSITES_GEO_POT'

for level in ['200hPa', '500hPa', '700hPa', '850hPa', '1000hPa']:
	for ch in ['TT','PP','PN']:
		for mes in ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']:

			path    = path + '/' + level + '/' + ch + '/' + mes
			folders = listdir(path)

			for name in folders:

				path2 = path + '/' + name
				if listdir(path2) != []:

					command1 = 'convert -delay 100 -loop 0  ' + path2 + '/*.png ' + path2 + '/myimage.gif'
					process1 = subprocess.Popen(command1.split(), stdout=subprocess.PIPE)

					output, error = process1.communicate()
					print error





