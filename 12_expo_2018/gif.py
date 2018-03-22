import subprocess
from os import listdir

path1 = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/12_expo_2018/COMPOSITES_GEO_POT'

for level in ['1000hPa']:
	for ch in ['TT','PP','PN']:
		for mes in ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']:

			path2    = path1 + '/' + level + '/' + ch + '/' + mes
			folders = listdir(path2)

			for name in folders:

				path3 = path2 + '/' + name
				if listdir(path2) != []:

					print level + '-' + ch + '-' + mes + '      ' + name
					command1 = 'convert -delay 100 -loop 0  ' + path3 + '/*.png ' + path3 + '/myimage.gif'
					process1 = subprocess.Popen(command1.split(), stdout=subprocess.PIPE)

					output, error = process1.communicate()
					print error






