import subprocess
import shutil

def generate_cpu_project(project) :
	f = open('template_cpu.vcxproj')
	text = f.read()
	f.close()
	text = text.replace("@name@", project[0])
	text = text.replace("@GUID@", project[1])
	f = open(project[0] + ".vcxproj", 'w')
	f.write(text)
	f.close

def generate_cuda_project(project) :
	f = open('template_cuda.vcxproj')
	text = f.read()
	f.close()
	text = text.replace("@name@", project[0])
	text = text.replace("@GUID@", project[1])
	f = open(project[0] + ".vcxproj", 'w')
	f.write(text)
	f.close


config='Release'
generate=True
build=True
copy=True

# build
subprocess.call('msbuild ../sqaodc/sqaodc.vcxproj /property:Configuration=' + config, shell=True)  

projects = []
projects.append(("cpu_formulas", "7099b44e-dba3-4d07-8824-622ed1de0570"))
projects.append(("cpu_dg_bf_searcher", "6068ceca-94ad-40f2-a0d5-d90f58c02e2e"))
projects.append(("cpu_dg_annealer", "973318d4-3110-4a4e-8508-97856ef046b7"))
projects.append(("cpu_bg_bf_searcher", "ac9eb479-e526-47b5-8b09-103874f1c1ae"))
projects.append(("cpu_bg_annealer", "ea2132df-2463-41b0-861b-d306c97bd42f"))


# generate projects
for project in projects:
	generate_cpu_project(project)

for project in projects:
	subprocess.call('msbuild ' + project[0] + '.vcxproj /property:Configuration=' + config, shell=True)  

#install
for project in projects:
	fn = project[0] + '.pyd'
	print project[0]
	shutil.copyfile('../x64/' + config + '/' + fn, '../../sqaodpy/sqaod/cpu/' + fn)

projects = []
projects.append(("cuda_device", "4002be82-f681-4b04-b569-e022c8fdce0d"))
projects.append(("cuda_formulas", "7099b44e-dba3-4d07-8824-622ed1de0570"))
projects.append(("cuda_dg_bf_searcher", "d7a73080-d71d-4e56-be64-8670ec777624"))
projects.append(("cuda_dg_annealer", "a7a2bb4a-0d92-4931-9d32-e7ecd9721de7"))
projects.append(("cuda_bg_bf_searcher", "a9611f72-4adb-4a91-9313-7a435e7e75d2"))
projects.append(("cuda_bg_annealer", "c87406bb-1d4e-496b-a1e4-77208e6dfc6a"))

# generate projects
for project in projects:
	generate_cuda_project(project)

for project in projects:
	subprocess.call('msbuild ' + project[0] + '.vcxproj /property:Configuration=' + config, shell=True)  

#install
for project in projects:
	fn = project[0] + '.pyd'
	print project[0]
	shutil.copyfile('../x64/' + config + '/' + fn, '../../sqaodpy/sqaod/cuda/' + fn)
