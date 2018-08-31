#!/usr/bin/env python3
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from obspy.core.utcdatetime import UTCDateTime as UTC
import code
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec

#things that should go into a class as defined variables:
#-mombins, make sure they're put in the correct functions that use them as well
#-percent of events to select for the bootstrapping instance
#-anything else I think of while I'm in the process of editing to make this code a class based code

#------------------------------------------------------------------------------------------------------------------------#
#setting up a class to make some of the constant variables a bit easier to manage
class reps:
	#setting up the default features
	def __init__(self):
		#setting default values
		self.magthresh = 0.2 #magnitude threshold
		self.locthresh = 0.015 #location uncertainty to allow
		self.depththresh = 0.1 #depth uncertainty to allow
		self.magconst = 1.5 #magnitude constant, 1.5 normally and 1.2 from Jess
		self.distthresh = 1. #distance threshold for selecting events for ratios plotting
		
		
		#setting up distance bins to use in ratios plotting
		self.distbins = np.array([0.,50.,100.,250.,500.,750.])
		
		
		
		
		
		
		
	#now going into the functions, and making sure that I save results to the class, as well as other relevant things I've thought of
	#------------------------------------------------------------------------------------------------------------------------#
	#defining main function
	def main(self):
		#opening pdf
		ppdf = PdfPages('parkfieldfigs.pdf')

		#running calculations
		self.readcat("ddparkfieldcat.csv")
		self.calctr()
		self.momtrrats()
		self.bootstrapmedians()
		self.weightedls()

		#plotting
		fig = self.plotruptimes()
		fig2 = self.plotmomtimes()
		fig3 = self.plotrats()
		fig4 = self.plotrats(zoom=True)
		plt.show()

		#saving to pdf and closing
		ppdf.savefig(fig)
		ppdf.savefig(fig2)
		ppdf.savefig(fig3)
		ppdf.close()
		plt.show()
		

	#defining function for testing throwing out earthquakes based on a rupture length threshold
	def test(self):
		self.readcat("ddparkfieldcat.csv")
		self.checkcalctr(rupthresh=0.03,returnvals=True)
		fig = self.plotruptimes()
		fig2 = self.plotmomtimes()
		self.momtrrats(ploton=False)
		fig3 = self.plotrats(distbins=True)
		plt.show()
		

	#------------------------------------------------------------------------------------------------------------------------#
	def codeint(self,localsvar,globalsvar):
		code.interact(local=dict(globalsvar, **localsvar))

	#------------------------------------------------------------------------------------------------------------------------#
	#calculating the distance between locations
	def distcalc(self,radolat,radolon,radstla,radstlo,cosolat,cosstla):
		#evla and evlo should be the original event locations
		#stla and stlo should be the event locations you want to calculate the distance to

		#converting points to radians
		#radstla = np.radians(stla)
		#radstlo = np.radians(stlo)
		#radolat = np.radians(evla)
		#radolon = np.radians(evlo)

		#using the haversine formula to calculate distance
		latang = np.divide(np.subtract(radstla,radolat),2.)
		lonang = np.divide(np.subtract(radstlo,radolon),2.)
		aval = (np.sin(latang)**2.0) +cosstla*cosolat*(np.sin(lonang))**2.0
		#dist = np.multiply(np.arcsin(np.sqrt(aval)),12742) #2x radius of earth
		dist = np.arcsin(np.sqrt(aval))*12742. #2x radius of earth
		#cval = np.multiply(np.arcsin(np.sqrt(aval)),2)

		#dist = np.multiply(np.around(6371.*cval,decimals=1),1000)
		#dist = np.multiply(cval,6371.)
		return dist

	#simpler version when I don't need speed
	def simpdistcalc(self,evla,evlo,stla,stlo):
		#evla and evlo should be the original event locations
		#stla and stlo should be the event locations you want to calculate the distance to

		#converting points to radians
		radstla = np.radians(stla)
		radstlo = np.radians(stlo)
		radolat = np.radians(evla)
		radolon = np.radians(evlo)

		#using the haversine formula to calculate distance
		latang = np.divide(np.subtract(radstla,radolat),2.)
		lonang = np.divide(np.subtract(radstlo,radolon),2.)
		aval = (np.sin(latang)**2.0) +np.cos(radstla)*np.cos(radolat)*(np.sin(lonang))**2.0
		cval = np.multiply(np.arcsin(np.sqrt(aval)),2.)

		#outputting distance
		dist = np.multiply(6371.*cval,1000.)
		return dist

	#------------------------------------------------------------------------------------------------------------------------#
	#reading in the actual eq catalogue
	def readcat(self,fname=None):
		from datetime import datetime as dt
		#setting default
		if fname is None:
			fname = "DDeditcat.csv"

		evid = []
		evdates = []
		evlats = []
		evlons = []
		evdeps = []
		evmags = []
		evmagtype = []
		with open(fname,'r') as f:
			reader = csv.reader(f,delimiter=',')
			for row in reader:
				evid.append(row[-1])
				evdates.append(dt.strptime(row[0],"%Y/%m/%d %H:%M:%S.%f"))
				evlats.append(float(row[1]))
				evlons.append(float(row[2]))
				evdeps.append(float(row[3]))
				evmags.append(float(row[4]))
				evmagtype.append(row[5])

		evid = np.array(evid)
		evdates = np.array(evdates)
		evlats = np.array(evlats)
		evlons = np.array(evlons)
		evdeps = np.array(evdeps)
		evmags = np.array(evmags)
		evmagtype = np.array(evmagtype)	


		#getting moments
		evmoms = np.multiply(np.power(10,np.add(np.multiply(evmags,self.magconst),16.0)),10.**-7) #using new constant from Jess, and the equation 
		#direct from Hanks and Kanamori

		#calculating earthquake rupture size
		#we assume its logarithmic with size, so M3 = 100m, M4 = 1km, M5 = 10km etc
		#calculate the rupture and then divide it by two to get the rupture RADIUS
		#note we're assuming that the earthquake is a circular rupture here
		#WELLS AND COPPERSMITH 1994 EQUATION
		#evrups = np.multiply(np.divide(np.power(10,np.divide(np.subtract(evmags,4.38),1.49)),2.),1000.) 
		#SIMPLISTIC EQUATION
		#evrups = np.divide(np.power(10,np.subtract(evmags,1)),2.)
		#ESHELBY CALCULATION ASSUMING STRESS DROP OF 10MPA
		tempsd = 10**7 #10MPa stress drop
		evrups = np.power(np.multiply(7./16.,np.divide(evmoms,tempsd)),1./3.)
		
		#calculating radian versions of lats and longs for speed up later
		radlats = np.radians(evlats)
		radlons = np.radians(evlons)
		#and also cosine of lats
		coslats = np.cos(radlats)
		

		evinfo = {"evid":evid,"evdates":evdates,"evlats":evlats,"evlons":evlons,"evdeps":evdeps,"evmags":evmags,"evmagtype":evmagtype,
				"evrups":evrups,"evmoms":evmoms,"radlats":radlats,"radlons":radlons,"coslats":coslats}

		self.evinfo = evinfo

	#------------------------------------------------------------------------------------------------------------------------#
	#selecting an area of the earthquake catalogue is I don't want to use the entire thing
	def selectarea(self,lons=None,lats=None,timelimit=None,bufftime=(6*30.*24*60*60)):
		"""
		selectarea(evinfo,lons,lats,timelimit=None,bufftime=None)
		lons, lats:	List of longitudes and latitudes of points along the area defined
		time limit:	Should be a time for which you don't want any earthquakes
				afterwards, in UTC format
		bufftime:	Buffer in seconds to put before the given time, currently differs at 
				6 months

		Function returns a new version of evinfo with only the selected earthquakes from the given area. Area can be any shape
		as long as the user provides the bounding corners.
		"""
		#importing functions
		from shapely.geometry import Point
		from shapely.geometry.polygon import Polygon 

		#first grabbing the time limit
		if timelimit is not None:
			#note this requires a specific format of date time string as UTC
			limtime = UTC(dt.strptime(timelimit,"%Y-%m-%dT%H:%M:%S.%fZ"))

			#calculating time difference
			difftime = np.subtract(limtime,self.evinfo["evdates"])

			#finding good times
			goodtimes = np.argwhere(difftime>bufftime).flatten()

		elif timelimit is None:
			goodtimes = np.arange(0,len(self.evinfo["evdates"])-1,1)

		#default to the Parkfield area
		if lons is None and lats is None:
			lons = [-120.85,-120.76,-120.19,-120.28]
			lats = [36.19,36.25,35.73,35.67]

		#creating the bounding box
		boxlons = np.array(lons)
		boxlats = np.array(lats)
		lons_lats_vect = np.column_stack((boxlons, boxlats)) # Reshape coordinates
		polygon = Polygon(lons_lats_vect) # create polygon

		#now need to check whether the locations are within the Parkfield area
		goodbox = []
		for i in range(len(self.evinfo["evlats"])):
			pt = Point(self.evinfo["evlons"][i],self.evinfo["evlats"][i])
			if polygon.contains(pt) is True:
				goodbox.append(i)
		goodbox = np.array(goodbox)

		#now identifying events which are both good in time and location
		goodlocs = np.argwhere(np.in1d(goodtimes,goodbox)).flatten()


		#selecting new catalogue
		evinfo2 = {"evid":self.evinfo["evid"][goodlocs],"evdates":self.evinfo["evdates"][goodlocs],"evlats":self.evinfo["evlats"][goodlocs],
			"evlons":self.evinfo["evlons"][goodlocs],"evdeps":self.evinfo["evdeps"][goodlocs],"evmags":self.evinfo["evmags"][goodlocs],
			"evmoms":self.evinfo["evmoms"][goodlocs],"evmagtype":self.evinfo["evmagtype"][goodlocs],"evrups":self.evinfo["evrups"][goodlocs],
			"radlats":self.evinfo["radlats"][goodlocs],"radlons":self.evinfo["radlons"][goodlocs],"coslats":self.evinfo["coslats"][goodlocs]}

		if len(goodlocs) == 0:
			print("Warning: No earthquakes available in that region")

		return evinfo2


	#------------------------------------------------------------------------------------------------------------------------#
	#plotting a map of the eqs
	def ploteqs(self):
		from mpl_toolkits.basemap import Basemap
		#first setting up the basemap of the appropriate area
		fig = plt.figure(1,figsize=(16,8))
		m = Basemap(projection='merc',llcrnrlat=np.min(self.evinfo["evlats"])-0.5,urcrnrlat=np.max(self.evinfo["evlats"])+0.5,\
        	    llcrnrlon=np.min(self.evinfo["evlons"])-0.5,urcrnrlon=np.max(self.evinfo["evlons"])+0.5,lat_ts=20,resolution='c',area_thresh=0.1)
		    #llcrnrlon=-131.319824,urcrnrlon=-126,lat_ts=20,resolution='c',area_thresh=0.1)
		m.drawcoastlines()
		m.fillcontinents(color='grey',lake_color='white',zorder=1)

		# draw parallels and meridians.
		m.drawparallels(np.arange(-90.,91.,0.1),labels=[True,False,False,False])
		m.drawmeridians(np.arange(-180.,181.,0.1),labels=[False,False,False,True])
		m.drawmapboundary(fill_color='white')

		#plotting on the events
		m.scatter(self.evinfo["evlons"],self.evinfo["evlats"],latlon=True,c=self.evinfo["evmags"],zorder=2)

		plt.colorbar()
		plt.show()

	#-----------------------------------------------------------------------------------------------------------------------#
	#plotting figure of earthquake locations in general across the Parkfield area in the same scheme as the Blanco fault plot
	def fig1plot(self):
		#FUNCTION HAS NOT BEEN OPTIMISED, BASED OFF OLD PROGRAM
		import matplotlib.dates as md
		from matplotlib.patches import Polygon
		from mpl_toolkits.basemap import Basemap

		#setting up pdf to save figure in
		pp = PdfPages('parkfieldfig1.pdf')

		#setting the font size to change labels to
		nfontsize = 14

		# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
		# are the lat/lon values of the lower left and upper right corners
		# of the map.
		# lat_ts is the latitude of true scale.
		# resolution = 'c' means use crude resolution coastlines.
		subplotgridsize = (4,3)

		fig = plt.figure(figsize=(10,8))
		ax1 = plt.subplot2grid(subplotgridsize,(0,0),colspan=4)
		ax = plt.subplot2grid(subplotgridsize,(1,1),colspan=2,rowspan=3)

		#setting up the map
		#m = Basemap(projection='merc',llcrnrlat=round(np.min(evinfo["evlats"])-0.2,1),urcrnrlat=round(np.max(evinfo["evlats"])+0.2,1),
        	#	llcrnrlon=round(np.min(evinfo["evlons"])-0.2,1),urcrnrlon=round(np.max(evinfo["evlons"])+0.2,1),lat_ts=20,resolution='c',area_thresh=0.1)
		m = Basemap(projection='merc',llcrnrlat=35.6,urcrnrlat=36.40000001,
        		llcrnrlon=-121.0000001,urcrnrlon=-120.2,lat_ts=20,resolution='c',area_thresh=0.1)
		m.drawcoastlines()
		m.fillcontinents(color='grey',lake_color='white',zorder=1)

		# draw parallels and meridians.
		m.drawparallels(np.arange(-90.,91.,0.2),labels=[True,False,False,False],fontsize=nfontsize)
		m.drawmeridians(np.arange(-180.,181.,0.2),labels=[False,False,False,True],fontsize=nfontsize,rotation=45)
		m.drawmapboundary(fill_color='white')


		m.ax = ax


		#------------------------------------------------------#
		print('Plotting eqs on map')
		#plotting event locations
		#Loading in events to plot on the map as well
		lonseq = self.evinfo["evlons"]
		latseq = self.evinfo["evlats"]
		magseq = self.evinfo["evmags"]
		dates = self.evinfo["evdates"]

		#Separating by magnitude
		locs0 = np.argwhere(magseq<1.0).flatten()
		locs1 = np.argwhere(np.logical_and(magseq>=1.0,magseq<2.0)).flatten()
		locs2 = np.argwhere(np.logical_and(magseq>=2.0,magseq<3.0)).flatten()
		locs3 = np.argwhere(np.logical_and(magseq>=3.0,magseq<4.0)).flatten()
		locs4 = np.argwhere(magseq>=4.0).flatten()

		#plotting earthquakes again
		m.scatter(lonseq[locs0],latseq[locs0],latlon=True,c='y',s=15,label='0.0 '+u'\u2264'+' M < 1.0',edgecolor='black',zorder=2)
		m.scatter(lonseq[locs1],latseq[locs1],latlon=True,c='b',s=25,label='1.0 '+u'\u2264'+' M < 2.0',edgecolor='black',zorder=3)
		m.scatter(lonseq[locs2],latseq[locs2],latlon=True, c='r',s=35,label='2.0 '+u'\u2264'+' M < 3.0',edgecolor='black',zorder=4)
		m.scatter(lonseq[locs3],latseq[locs3],latlon=True, c='g',s=45,label='3.0 '+u'\u2264'+' M < 4.0',edgecolor='black',zorder=5)
		m.scatter(lonseq[locs4],latseq[locs4],latlon=True, c='c',s=55,label='4.0 '+u'\u2264'+' M',edgecolor='black',zorder=6)

		#Finally, adding a legend
		ax.legend(bbox_to_anchor=(-0.25,1.05),fontsize=nfontsize)
		plt.setp(ax.get_xticklabels(),fontsize=nfontsize)

		#Adding a map scale
		m.drawmapscale(-120.7, 35.7, -120.7, 35.7, 50, barstyle='fancy') 

		#--------------------------------------------------------------------------------------------------------------#
		#PLOTTING THE EARTHQUAKES OVER TIME ON THE Parkfield FAULT

		print('Plotting eqs over time')

		fig.set_facecolor('None')

		#Defining the fault
		stlat = 36.23
		stlon = -120.79
		enlat = 35.69
		enlon = -120.25

		#Forming the line
		coeffs = np.polyfit([stlon,enlon],[stlat,enlat],1)

		#So equation of the fault line I've set is y = grad*x + c
		#Generating all the points along this line in increments of 0.0001 lat/lon
		linelons = np.arange(stlon,enlon,0.0001)
		linelats = np.add(np.multiply(linelons,coeffs[0]),coeffs[1])

		#getting location along the line in km
		lineloc = np.sqrt(np.add(np.power(np.abs(linelats-stlat),2.0),np.power(np.abs(linelons-stlon),2.0)))*111.32


		#Letting python see dates are dates
		projlocs = []
		#Projecting the points onto the line
		for j in range(len(magseq)):
			#totdiff = []
			latdiff = np.abs(np.subtract(latseq[j],linelats))
			londiff = np.abs(np.subtract(lonseq[j],linelons))
			totdiff = np.sqrt(np.add(np.power(latdiff,2.0),np.power(londiff,2.0)))

			#Finding location for this earthquake to be projected too
			index = np.argwhere(totdiff==np.min(totdiff))[0]
			projlocs.append(lineloc[index])


		projlocs = np.array(projlocs)

		#With final projected locations the earthquakes can be plotted with different marker size
		ax1.plot_date(dates[locs4],projlocs[locs4],markersize=10,c='c',markeredgecolor='k',zorder=6)
		ax1.plot_date(dates[locs3],projlocs[locs3],markersize=8,c='g',markeredgecolor='k',zorder=5)
		ax1.plot_date(dates[locs2],projlocs[locs2],markersize=6,c='r',markeredgecolor='k',zorder=4)
		ax1.plot_date(dates[locs1],projlocs[locs1],markersize=5,c='b',markeredgecolor='k',zorder=3)
		ax1.plot_date(dates[locs0],projlocs[locs0],markersize=4,c='y',markeredgecolor='k',zorder=2)


		ax1.xaxis.tick_top()
		ax1.set_ylabel('Distance along strike (km)',fontsize=nfontsize)
		ax1.xaxis.set_label_position('top')
		ax1.set_ylim(0,np.max(projlocs))
		ax1.tick_params(axis='both',labelsize=nfontsize)
		ax1.grid(True)
		ax1.invert_yaxis()


		###-----------------------------------------------------------------------------------#####
		#plotting small zoom in inset
		print('Plotting small inset')
		ax2 = plt.subplot2grid(subplotgridsize,(2,0),colspan=1,rowspan=2)


		#globe with blanco area
		m2 = Basemap(projection='merc',llcrnrlat=32,urcrnrlat=38,
        		    llcrnrlon=-124,urcrnrlon=-118,lat_ts=20,resolution='h',area_thresh=0.1)
		m2.drawcoastlines()
		m2.drawstates()
		m2.fillcontinents(color='gray',lake_color='white')

		# draw parallels and meridians.
		m2.drawparallels(np.arange(-90.,91.,2.),labels=[True,False,False,False],fontsize=nfontsize)
		m2.drawmeridians(np.arange(-180.,181.,2.),labels=[False,False,False,True],fontsize=nfontsize,rotation=45)
		m2.drawmapboundary(fill_color='white')
		m2.ax = ax2
		m2.readshapefile('tectonic_plates_files/PB2002_boundaries', 
                		name='tectonic_plates', drawbounds=True,
                		color='red',linewidth = 1.0)

		#Drawing rectangle defining fault area on map
		lonspk = [-120.85,-120.76,-120.19,-120.28]
		latspk = [36.19,36.25,35.73,35.67]
		xpk, ypk = m2(lonspk, latspk)
		xypk = zip(xpk,ypk)
		poly = Polygon(xypk,facecolor='none',edgecolor='green',linewidth=2.0,zorder=10)
		plt.gca().add_patch(poly)

		#adding labels to show California state and the Pacific Ocean

		#plt.tight_layout()
		plt.subplots_adjust(wspace=0.05)
		pp.savefig(fig,bbox_inches='tight',facecolor='None',transparent=True)
		plt.savefig('parkfieldfig1.png',dpi=1000)
		pp.close()
		plt.show()

	#-----------------------------------------------------------------------------------------------------------------------#
	#plotting earthquakes in a specific area, with radii circles around each earthquake
	def ploteqrups(self,lats=None,lons=None):
		"""
		lats:	Latitudes of the corners of the box, first value should be the lower left latitude, second the upper right
		lons:	Longitudes with same format as lats
		"""
		from mpl_toolkits.basemap import Basemap
		import matplotlib as mpl
		import matplotlib.cm as cm
		from matplotlib import gridspec
		from matplotlib.patches import Circle,Wedge

		#if lats and lons are not given then use defaults
		if lats is None and lons is None:
			#first default area has a few too many sequences
			#lats = [35.924,35.926]
			#lons = [-120.473,-120.471]

			#second default area
			lats = [36.147,36.149]
			lons = [-120.727,-120.724]

			#zoom in of second set of repeaters
			lats = [36.148,36.149]
			lons = [-120.726,-120.724]


		#first setting up the basemap of the appropriate area
		fig = plt.figure(1,figsize=(16,8))
		m = Basemap(projection='merc',llcrnrlat=lats[0],urcrnrlat=lats[1],\
        	    llcrnrlon=lons[0],urcrnrlon=lons[1],lat_ts=20,resolution='c',area_thresh=0.1)
		m.drawcoastlines()
		m.fillcontinents(color='lightgray',lake_color='white',zorder=1)

		# draw parallels and meridians.
		m.drawparallels(np.arange(lats[0],lats[1],0.0005),labels=[True,False,False,False])
		m.drawmeridians(np.arange(lons[0],lons[1],0.001),labels=[False,False,False,True])
		m.drawmapboundary(fill_color='white')



		#repeating the longitudes and latitudes to make a square box rather than just two points
		nlats = [lats[0],lats[1],lats[1],lats[0]]
		nlons = [lons[0],lons[0],lons[1],lons[1]]


		#identifying events that are within the set box
		evinfo2 = self.selectarea(nlons,nlats)

		#-----------IDENTIFYING SEQUENCES---------------------------#	
		#now we need to know which of these form sequences
		trinfo2 = self.calctr(evinfo=evinfo2,returnvals=True)

		#identifying sequences in the dataset
		allseqlocs = self.identseq(trinfo2)


		#-------------------PLOTTING SEQUENCES-------------------------------------------------------------#
		#so now we can plot the events on with colours depending on the sequences they belong to
		#in order to get the colouring scheme I want, I'm actually going to have to use Circle
		#and Wedge classes

		#first setting up a list of colours for each of the sequences to be
		norm = mpl.colors.Normalize(vmin=0, vmax=len(allseqlocs))
		cmap = cm.Set1
		#cmap = cm.Dark2
		cmscal = cm.ScalarMappable(norm=norm, cmap=cmap)
		colors = cmscal.to_rgba(range(0,len(allseqlocs)))

		#now we want to plot each earthquake with the color that corresponds to its sequence
		#and make the circle the radius of the earthquake rupture
		#but first we identify points that appear in multiple sequences
		#flattening sequence list
		flatseqs = [item for sublist in allseqlocs for item in sublist]
		#finding the max index
		maxidx = np.max(flatseqs)+1

		#now going through each index and counting the number of times it appears in sequences
		#also saving the index of which sequence it appears in
		appearseqs = [[] for i in range(maxidx)]
		#for each eq
		for i in range(maxidx):
			#for each sequence
			for j in range(len(allseqlocs)):
				#test if the earthquake is in that sequence
				ix = [n for n, x in enumerate(allseqlocs[j]) if i == x]

				#if it is then add one to its number and save the sequences it appears in
				if len(ix) != 0:
					appearseqs[i].append(j)



		#getting the current axes
		ax = plt.gca()

		#plotting the location of repeaters with circles indicating the event rupture size
		self.plotreploc(m,ax,evinfo2,appearseqs,colors)
		#SHOULD ADD PLOTTING OF EARTHQUAKES THAT DON'T APPEAR IN A SEQUENCE
		#JUST PLOT THEM AS BLACK OR SOMETHING

		#make up a set of lines for plotting the legend
		lines =[]
		for ij in range(len(allseqlocs)):
			lines.append(plt.plot(0,0,c=colors[ij],label='Sequence '+str(ij+1)))

		#creating legend
		plt.legend(loc='center right',bbox_to_anchor=(1.4,0.5))



		#-------------------------------------------------------------------------------#
		#next step is to calculate time to next eq moment for each eq pair and plot coloured by the relevant eq sequence
		#extracting
		evrups = evinfo2["evrups"]
		savrups = trinfo2["savrups"]
		savmoms = trinfo2["savmoms"]
		savtims = trinfo2["savtims"]
		savmags = trinfo2["savmags"]
		

		#taking log of moments to make it easier to bin
		savmomslog = np.log10(savmoms)
		locs = np.where(np.isfinite(savmoms)==True)

		#setting up figure
		fig40 = plt.figure(40,figsize=(16,8))
		gs = gridspec.GridSpec(2,1,height_ratios=[3,1])

		#plotting the tr versus eq rupture size subplot
		ax0 = plt.subplot(gs[0])

		#binning the event moments
		stval = round(np.min(savmomslog)-0.05,1)
		enval = round(np.max(savmomslog)+0.05,1)
		mombins = np.arange(stval,enval,0.1)

		#getting the mode of each bin
		medtims = self.modefunc(savmomslog,savtims,mombins)

		#finding locations where medtims is not nan
		locs2 = np.where(np.isfinite(medtims)==True)
		mombinspow = np.power(10,mombins[locs2])


		#plotting the moments and times as coloured by the eq sequence it belongs to

		ix2 = [n for n, x in enumerate(appearseqs) if len(x) !=0]
		#looping through each earthquake
		for val in ix2:
			for jk in range(len(trinfo2["evtclist"])):
				if val == trinfo2["evtclist"][jk]:
					ax0.plot(np.log10(savmoms[jk]),np.log10(savtims[jk]),'+',c=colors[appearseqs[val][0]])

					#setting radius
					radius = 0.025



					#first checking if the earthquake is only in one sequence	
					if len(appearseqs[val]) == 1:
						#then plot a circle
						#now plotting circle
						circle1 = Circle((np.log10(savmoms[jk]), np.log10(savtims[jk])), radius, color=colors[appearseqs[val][0]],fill=False,zorder=3)
						ax0.add_patch(circle1)

					#and if it is in multiple sequences
					elif len(appearseqs[val]) > 1:
						#then count how many sequences it appears in and plot wedges
						nseqs = len(appearseqs[val])

						#getting angles
						angle = 0
						#getting iteration
						angit = 360./nseqs

						#generating angles and plotting wedges
						thetas = []
						for ik in range(nseqs):
							theta1 = angle+(ik*angit)
							theta2 = angle+((ik+1)*angit)
							pltwedge = Wedge([np.log10(savmoms[jk]),np.log10(savtims[jk])],radius,theta1,theta2,fill=False,edgecolor=colors[appearseqs[val][ik]],zorder=2)
							ax0.add_patch(pltwedge)




		#doing rest of plotting
		#ax0.plot(mombinspow,medtims[locs2],'ks',markersize=20,label='median')
		ax0.set_xlabel('Earthquake moment on log10 scale (N m)')
		#ax0.set_xscale('log')
		ax0.set_ylabel('Time to the next earthquake within rupture size on log10 scale (s)')
		#ax0.set_yscale('log')

		ax0.set_ylim([np.min(np.around(np.log10(savtims),1))-0.2,np.max(np.around(np.log10(savtims),1))+0.2])
		ax0.set_xlim([np.min(np.around(np.log10(savmoms),1))-0.2,np.max(np.around(np.log10(savmoms),1))+0.2])

		#plotting the relation you expect given M0^1/3 and M0^1/6
		tr13 = np.multiply(np.power(mombinspow,1./3.),np.divide(np.median(medtims[locs2]),np.median(np.power(mombinspow,1./3.))))
		tr16 = np.multiply(np.power(mombinspow,1./6.),np.divide(np.median(medtims[locs2]),np.median(np.power(mombinspow,1./6.))))
		tr112 = np.multiply(np.power(mombinspow,1./12.),np.divide(np.median(medtims[locs2]),np.median(np.power(mombinspow,1./12.))))


		#plotting the relations you expect
		#ax0.plot(mombins[locs2],np.log10(tr13),'g--',label='M0^1/3')
		#ax0.plot(mombins[locs2],np.log10(tr16),'r--',label='M0^1/6')
		#ax0.plot(mombins[locs2],np.log10(tr112),'c--',label='M0^1/12')

		#make up a set of lines for plotting the legend
		lines =[]
		for ij in range(len(allseqlocs)):
			lines.append(ax0.plot(0,0,c=colors[ij],label='Sequence '+str(ij+1)))


		ax0.legend(loc='lower right',bbox_to_anchor=(1.1,0.5))

		#now plotting the histogram of how many values are going into each median
		ax1 = plt.subplot(gs[1])
		bins = np.arange(np.min(savmomslog[locs])-0.05,np.max(savmomslog[locs])+0.15,0.1)
		nums,bns,c = ax1.hist(savmomslog[locs],bins,edgecolor="black",linewidth=1.0)
		ax1.set_xlabel('Earthquake moment (log10 scale)')

		#calculating the gradient in normal and log space
		gradient, blah = np.polyfit(mombins[locs2],medtims[locs2],1)
		gradientlog, blah = np.polyfit(mombins[locs2],np.log10(medtims[locs2]),1)
		print("Linear gradient: "+str(gradient))
		print("Log-Linear gradient: "+str(gradientlog))
		
		
		

		#----------------------------------------------------------------------------------------------------------------#
		#DEFINING THE FAULT PROJECTION
		#CURRENTLY DEFAULTS TO THE PARKFIELD SEGMENT
		#Defining the fault
		stlat = 36.23
		stlon = -120.79
		enlat = 35.69
		enlon = -120.25

		#Forming the line
		coeffs = np.polyfit([stlon,enlon],[stlat,enlat],1)

		#So equation of the fault line I've set is y = grad*x + c
		#Generating all the points along this line in increments of 0.0001 lat/lon
		newlons = np.arange(stlon,enlon,0.0001)
		newlats = np.add(np.multiply(newlons,coeffs[0]),coeffs[1])

		#getting location along the line in km
		lineloc = np.sqrt(np.add(np.power(np.abs(newlats-stlat),2.0),np.power(np.abs(newlons-stlon),2.0)))*111.32


		#projecting each earthquake in the dataset onto the fault
		evlats = evinfo2["evlats"]
		evlons = evinfo2["evlons"]
		projlats = []
		projlons = []
		projlocs = []
		for ik in range(len(evlats)):
			#calculate the position to project this earthquake onto the fault
			totdiff = np.sqrt(np.add(np.power(np.abs(evlats[ik]-newlats),2.),np.power(np.abs(evlons[ik]-newlons),2.)))

			#finding the minimum difference to give the location to project it to
			locloc = np.argwhere(totdiff==np.min(totdiff)).flatten()

			#saving the location to project it onto
			projlats.append(newlats[locloc])
			projlons.append(newlons[locloc])
			projlocs.append(lineloc[locloc])

		#----------------------------------------------------------#
		#now plotting the earthquakes onto the fault plane against depth
		#and also plotting their radii

		#first setting up a list of colours for each of the sequences to be
		norm = mpl.colors.Normalize(vmin=0, vmax=len(allseqlocs))
		cmap = cm.Set1
		#cmap = cm.Dark2
		cmscal = cm.ScalarMappable(norm=norm, cmap=cmap)
		colors = cmscal.to_rgba(range(0,len(allseqlocs)))


		#setting up the figure
		fig30 = plt.figure(30,figsize=(10,8))
		ax = plt.gca()

		#first saving some time by just finding which earthquakes appear in a sequence
		ix2 = [n for n, x in enumerate(appearseqs) if len(x) !=0]
		#looping through each earthquake
		for val in ix2:
			#plotting a plus to show the centre of each earthquake
			plt.plot(projlocs[val],evinfo2["evdeps"][val],'+',c=colors[appearseqs[val][0]])

			#getting the earthquake rupture radius in km
			radius = 1./(evinfo2["evrups"][val])

			#first checking if the earthquake is only in one sequence	
			if len(appearseqs[val]) == 1:
				#then plot a circle
				#now plotting circle
				circle1 = Circle((projlocs[val], evinfo2["evdeps"][val]), radius, color=colors[appearseqs[val][0]],fill=False,zorder=3)
				ax.add_patch(circle1)

			#and if it is in multiple sequences
			elif len(appearseqs[val]) > 1:
				#then count how many sequences it appears in and plot wedges
				nseqs = len(appearseqs[val])

				#getting angles
				angle = 0
				#getting iteration
				angit = 360./nseqs

				#generating angles and plotting wedges
				thetas = []
				for jk in range(nseqs):
					theta1 = angle+(jk*angit)
					theta2 = angle+((jk+1)*angit)
					pltwedge = Wedge([projlocs[val][0],evinfo2["evdeps"][val]],radius,theta1,theta2,fill=False,edgecolor=colors[appearseqs[val][jk]],zorder=2)
					ax.add_patch(pltwedge)

		#make up a set of lines for plotting the legend
		lines =[]
		for ij in range(len(allseqlocs)):
			lines.append(plt.plot(0,0,c=colors[ij],label='Sequence '+str(ij+1)))



		#setting the limits to plot as based off available locations
		mindep = np.min(evinfo2["evdeps"])-0.1
		maxdep = np.max(evinfo2["evdeps"])+0.1
		minloc = np.min(projlocs)-0.15
		maxloc = np.max(projlocs)+0.15
		plt.ylim(mindep,maxdep)
		plt.xlim(minloc,maxloc)

		#flipping y axis
		plt.gca().invert_yaxis()

		#labels
		plt.ylabel('Depth (km)')
		plt.xlabel('Distance SE along Parkfield fault (km)')


		#creating legend
		plt.legend(loc='center right',bbox_to_anchor=(1.,0.5))
		
		#showing plots after running the function
		plt.show()

	
	#---------------------------------------------------------------------------------------------------------------------------------------#
	#small function to plot positions of repeaters
	def plotreploc(self,m,ax,evinfo2,appearseqs,colors):
		from matplotlib.patches import Circle,Wedge

		#pulling out longitudes and latitudes
		evlats = evinfo2["evlats"]
		evlons = evinfo2["evlons"]

		#now going through each earthquake location and plotting it with wedges for eqs in multiple sequences
		#and circles for earthquakes just in one sequence
		#first saving some time by just finding which earthquakes appear in a sequence
		ix2 = [n for n, x in enumerate(appearseqs) if len(x) !=0]
		#looping through each earthquake
		for val in ix2:
			#get coordinates on map
			x,y=m(evlons[val],evlats[val])

			#plotting a plus to show the centre of each earthquake
			m.plot(x,y,'+',c=colors[appearseqs[val][0]])

			#getting the earthquake rupture radius in degrees
			radius = 1./(111.32/(evinfo2["evrups"][val]/1000.))

			#applying shift
			x2,y2 = m(evlons[val],evlats[val]+radius)

			#first checking if the earthquake is only in one sequence	
			if len(appearseqs[val]) == 1:
				#then plot a circle
				#now plotting circle
				circle1 = Circle((x, y), y2-y, color=colors[appearseqs[val][0]],fill=False,zorder=3)
				ax.add_patch(circle1)

			#and if it is in multiple sequences
			elif len(appearseqs[val]) > 1:
				#then count how many sequences it appears in and plot wedges
				nseqs = len(appearseqs[val])

				#getting angles
				angle = 0
				#getting iteration
				angit = 360./nseqs

				#generating angles and plotting wedges
				thetas = []
				for jk in range(nseqs):
					theta1 = angle+(jk*angit)
					theta2 = angle+((jk+1)*angit)
					pltwedge = Wedge([x,y],y2-y,theta1,theta2,fill=False,edgecolor=colors[appearseqs[val][jk]],zorder=2)
					ax.add_patch(pltwedge)
					
					
					
					
	#-----------------------------------------------------------------------------------------------------------------------#
	#function to identify sequences
	def identseq(self,trinfo):
		#looping through each of the first events
		#setting up empty array for saving each of the sequences in
		allseqlocs = []
		evtcs = trinfo["evtclist"]
		glocs=  trinfo["gloclist"]

		#looping through each pair of eqs
		for i in range(len(evtcs)):
			#setting up list to save a sequence in
			seqlocs = []
			seqlocs.append(evtcs[i])
			seqlocs.append(glocs[i])

			#temporary check variable for looking for the next value in the sequence
			tempcheck = glocs[i]
			locs = [10,10]

			#while the sequence continues, keep checking for the sequence to continue
			while len(locs) != 0:
				#looking for next continuation of sequence
				locs = np.argwhere(evtcs==tempcheck).flatten()
				if len(locs) != 0:
					seqlocs.append(glocs[locs[0]])
					tempcheck = glocs[locs[0]]

			#save the whole sequence to the array
			#but first cheching the sequence isn't a subset of a sequence we've already identified
			dontsave = False
			for j in range(len(allseqlocs)):
				if set(seqlocs).issubset(allseqlocs[j]):
					dontsave = True

			#and only saving if it isn't a subset of a previous result
			if dontsave is False:
				allseqlocs.append(seqlocs)

		return allseqlocs			





	#-----------------------------------------------------------------------------------------------------------------------#
	#calculating the time to next earthquake within one radii
	#version of this function where I only use the next eq time to generate the recurrence
	#as in for each event I identify the next event within the rupture length and save the time to that next event
	#in the other function I was taking the time difference between all events
	#also added option to filter by the magnitude limit as well
	def calctr(self,returnvals=False,evinfo=None):
		#Function works in metres

		#extracting for ease of use
		if evinfo is None:
			evlats = self.evinfo["evlats"]
			evlons = self.evinfo["evlons"]
			evdeps = self.evinfo["evdeps"]
			evdates = self.evinfo["evdates"]
			evrups = self.evinfo["evrups"]
			evmags = self.evinfo["evmags"]
			evmoms = self.evinfo["evmoms"]
			radlats = self.evinfo["radlats"]
			radlons = self.evinfo["radlons"]
			coslats = self.evinfo["coslats"]
		else:
			evlats = evinfo["evlats"]
			evlons = evinfo["evlons"]
			evdeps = evinfo["evdeps"]
			evdates = evinfo["evdates"]
			evrups = evinfo["evrups"]
			evmags = evinfo["evmags"]
			evmoms = evinfo["evmoms"]
			radlats = evinfo["radlats"]
			radlons = evinfo["radlons"]
			coslats = evinfo["coslats"]
			
			
			
		#getting km version of evrups
		evrupskm= np.divide(evrups,1000.)
		

		#setting up saving arrays
		savrups = []
		savtims = []
		sav1mags = []
		savmoms = []
		sav1lats = []
		sav1lons = []
		sav1dates = []
		pradlats = []
		pradlons = []
		pcoslats = []
		sav2mags = []
		sav2lats = []
		sav2lons = []
		sav2dates = []
		evtclist = []
		gloclist = []


		#so now for each earthquake we need to find the time to the next eq within the rupture radius
		#and earthquakes have to be within 1 magnitude unit
		for evtc in range(len(evrups)-1):
			#first calculating the distance between this event and all the others in the catalogue
			dists = self.distcalc(radlats[evtc],radlons[evtc],radlats[evtc+1:],radlons[evtc+1:],coslats[evtc],coslats[evtc+1:]) #remaining eqs in catalogue

			#now finding locations where the distance is below the rupture radius
			#and excluding the case where its comparing an event location with itself
			#ADDED IN 15M UNCERTAINTY IN THE CATALOGUE LOCATIONS
			glocsdist = np.add(np.argwhere(dists<=evrupskm[evtc]+self.locthresh).flatten(),evtc+1) # for case where I use all ev locs

			#checking that the events are within 100m vertically as well
			glocsdep = np.add(np.argwhere(np.abs(np.subtract(evdeps[evtc+1:],evdeps[evtc]))<self.depththresh).flatten(),evtc+1)


			#finding events within distance and depth threshold
			glocs1 = glocsdep[np.in1d(glocsdep,glocsdist)]


			#finding locations where the magnitude is within 0.1 units
			glocs2 = np.add(np.argwhere(np.abs(evmags[evtc+1:]-evmags[evtc])<=self.magthresh).flatten(),evtc+1)


			#only accepting the earthquakes that fulfil depth, dist and mag requirements
			glocs = glocs1[np.in1d(glocs1,glocs2)]


			#only taking the next earthquake within the rupture length
			try:
			
				#now with these locations calculate the difference in seconds between the eq times
				times  = np.subtract(evdates[glocs[0]],evdates[evtc])


				#saving the values which I plot
				#for jj in range(len(times)):
				savrups.append(evrups[evtc])
				savmoms.append(evmoms[evtc])
				savtims.append(times.total_seconds())
				sav1mags.append(evmags[evtc])
				sav1lats.append(evlats[evtc])
				sav1lons.append(evlons[evtc])
				sav1dates.append(evdates[evtc])
				pradlats.append(radlats[evtc])
				pradlons.append(radlons[evtc])
				pcoslats.append(coslats[evtc])
				sav2mags.append(evmags[glocs[0]])
				sav2lats.append(evlats[glocs[0]])
				sav2lons.append(evlons[glocs[0]])
				sav2dates.append(evdates[glocs[0]])
				evtclist.append(evtc)
				gloclist.append(glocs[0])

			except:
				continue

		



		savrups = np.array(savrups)
		savtims = np.array(savtims)
		savmags = np.array(sav1mags)
		savlats = np.array(sav1lats)
		savlons = np.array(sav1lons)
		pradlats = np.array(pradlats)
		pradlons = np.array(pradlons)
		pcoslats = np.array(pcoslats)

		#now grabbing the median times for each rupture radius
		eachrup = np.unique(savrups[np.isfinite(savrups)])

		#binning the rupture lengths differently
		stval = round(np.log10(np.min(savrups))-0.05,1)
		enval = round(np.log10(np.max(savrups))+0.05,1)
		rupbins = np.arange(stval,enval,0.1)



		#log10 of savrups
		savrupslog = np.round(np.log10(savrups),1)

		#finding ruplengths in the bins
		#version saving the mode rather than the median
		medtims = self.modefunc(savrupslog,savtims,rupbins)

		trinfo = {"evtclist":evtclist,"gloclist":gloclist,"savrups":savrups, "savmoms":savmoms, "savtims":savtims, "savmags":savmags,
			"savlats":savlats,"savlons":savlons,"eachrup":eachrup, "medtims":medtims, "rupbins":rupbins,"radlats":pradlats,
			"radlons":pradlons,"coslats":pcoslats}
		
		if returnvals is True:
			return trinfo
		else:
			self.trinfo = trinfo
		
	#-----------------------------------------------------------------------------------------------------------------------------------#
	#version of calctr where all events with rupture lengths below 50m are eliminated and we look for events within 20m
	def checkcalctr(self,rupthresh=0.05,returnvals=False):
		#savrups, savtims, eachrup, medtims = o.calctr(evinfo)
		#Function works in metres

		#extracting for ease of use
		evlats = self.evinfo["evlats"]
		evlons = self.evinfo["evlons"]
		evdeps = self.evinfo["evdeps"]
		evdates = self.evinfo["evdates"]
		evrups = self.evinfo["evrups"]
		evmags = self.evinfo["evmags"]
		evmoms = self.evinfo["evmoms"]
		radlats = self.evinfo["radlats"]
		radlons = self.evinfo["radlons"]
		coslats = self.evinfo["coslats"]
		
		#getting km version of evrups
		evrupskm= np.divide(evrups,1000.)

		

		#setting up saving arrays
		savrups = []
		savtims = []
		savmoms = []
		sav1mags = []
		sav1lats = []
		sav1lons = []
		sav1dates = []
		pradlats = []
		pradlons = []
		pcoslats = []
		sav2mags = []
		sav2lats = []
		sav2lons = []
		sav2dates = []
		evtclist = []
		gloclist = []
		for evtc in range(len(evrups)-1):
			print(str(evtc+1)+' of '+str(len(evrups)-1))
			#first calculating the distance between this event and all the others in the catalogue
			dists = self.distcalc(radlats[evtc],radlons[evtc],radlats[evtc+1:],radlons[evtc+1:],coslats[evtc],coslats[evtc+1:]) #remaining eqs in catalogue

			#now finding locations where the distance is below the rupture radius
			#looking within 20m
			glocsdist = np.add(np.argwhere(dists<=evrupskm[evtc]+0.015).flatten(),evtc+1) # for case where I use all ev locs

			#checking that the events are within 100m vertically as well
			glocsdep = np.add(np.argwhere(np.abs(np.subtract(evdeps[evtc+1:],evdeps[evtc]))<0.1).flatten(),evtc+1)


			#finding events within distance and depth threshold
			glocs1 = glocsdep[np.in1d(glocsdep,glocsdist)]


			#finding locations where the magnitude is within 0.1 units
			glocs2 = np.add(np.argwhere(np.abs(evmags[evtc+1:]-evmags[evtc])<=self.magthresh).flatten(),evtc+1)


			#only accepting the earthquakes that fulfil depth, dist and mag requirements
			glocs = glocs1[np.in1d(glocs1,glocs2)]


			#only taking the next earthquake within the rupture length
			try:
			#if len(glocs) != 0:
				#glocs = np.array([glocs[0]])

				#now with these locations calculate the difference in seconds between the eq times
				times  = np.subtract(evdates[glocs[0]],evdates[evtc])

				#only saving results if the earthquake rupture length is greater than 50m
				if evrupskm[evtc] >= rupthresh:
					#saving the values which I plot
					#for jj in range(len(times)):
					savrups.append(evrups[evtc])
					savmoms.append(evmoms[evtc])
					savtims.append(times.total_seconds())
					sav1mags.append(evmags[evtc])
					sav1lats.append(evlats[evtc])
					sav1lons.append(evlons[evtc])
					sav1dates.append(evdates[evtc])
					pradlats.append(radlats[evtc])
					pradlons.append(radlons[evtc])
					pcoslats.append(coslats[evtc])
					sav2mags.append(evmags[glocs[0]])
					sav2lats.append(evlats[glocs[0]])
					sav2lons.append(evlons[glocs[0]])
					sav2dates.append(evdates[glocs[0]])
					evtclist.append(evtc)
					gloclist.append(glocs[0])

			except:
				continue


		savrups = np.array(savrups)
		savtims = np.array(savtims)
		savmags = np.array(sav1mags)
		savlats = np.array(sav1lats)
		savlons = np.array(sav1lons)
		pradlats = np.array(pradlats)
		pradlons = np.array(pradlons)
		pcoslats = np.array(pcoslats)

		#now grabbing the median times for each rupture radius
		eachrup = np.unique(savrups[np.isfinite(savrups)])

		#binning the rupture lengths differently
		stval = round(np.log10(np.min(savrups))-0.05,1)
		enval = round(np.log10(np.max(savrups))+0.05,1)
		rupbins = np.arange(stval,enval,0.1)

		#log10 of savrups
		savrupslog = np.round(np.log10(savrups),1)

		#finding ruplengths in the bins
		#version saving the mode rather than the median
		medtims = self.modefunc(savrupslog,savtims,rupbins)

		trinfo = {"evtclist":evtclist,"gloclist":gloclist,"savrups":savrups, "savmoms":savmoms, "savtims":savtims, "savmags":savmags,"savlats":savlats, 
			"savlons":savlons,"eachrup":eachrup, "medtims":medtims, "rupbins":rupbins,"radlats":pradlats,"radlons":pradlons,"coslats":pcoslats}

		if returnvals is True:
			return trinfo
		else:
			self.trinfo = trinfo
	#-----------------------------------------------------------------------------------------------------------------------------------#
	#function allowing import of trinfo from csv file
	#OLD VERSION, LIKELY DOESN'T WORK WITH THE NEW CLASS
	def importtrinfo(self,maglim=True):
		#creating empty lists
		evtclist = []
		gloclist = []
		savrups = []
		savtims = []
		savmags = []
		savlats = []
		savlons = []

		#setting csv file name depending on whether to look at data with magnitude limit
		if maglim is True:
			csvname = 'ddratsmaglim.csv'
		elif maglim is False:
			csvname = 'ddratsnomaglim.csv'

		#importing from the csv file
		with open(csvname,'r') as f:
			reader = csv.reader(f,delimiter='\t')
			for row in reader:
				evtclist.append(int(row[0]))
				gloclist.append(int(row[1]))
				savlats.append(float(row[2]))
				savlons.append(float(row[3]))
				savmags.append(float(row[4]))
				savrups.append(float(row[5]))
				savtims.append(float(row[6]))

		#turning them all into arrays
		evtclist = np.array(evtclist)
		gloclist = np.array(gloclist)
		savlats = np.array(savlats)
		savlons = np.array(savlons)
		savmags = np.array(savmags)
		savrups = np.array(savrups)
		savtims = np.array(savtims)

		#calculating eachrup, medtims and rupbins
		#now grabbing the median times for each rupture radius
		eachrup = np.unique(savrups[np.isfinite(savrups)])

		#binning the rupture lengths differently
		stval = round(np.log10(np.min(savrups))-0.05,1)
		enval = round(np.log10(np.max(savrups))+0.05,1)
		rupbins = np.arange(stval,enval,0.1)

		#log10 of savrups
		savrupslog = np.round(np.log10(savrups),1)

		#getting medtims
		medtims = self.modefunc(savrupslog,savtims,rupbins)

		#putting them into the dictionary
		trinfo = {"evtclist":evtclist, "gloclist":gloclist,"savrups":savrups, 
			"savtims":savtims, "savmags":savmags,"savlats":savlats, "savlons":savlons,
			"eachrup":eachrup, "medtims":medtims, "rupbins":rupbins}

		self.trinfo = trinfo

	#-----------------------------------------------------------------------------------------------------------------------------------#
	#comparing the moment and recurrence time ratios for each earthquake pair
	def momtrrats(self,ploton=False,trinfo=None,returnvals=False):
		
		#pulling out some results I'm going to use
		if trinfo is None:
			savtims = self.trinfo["savtims"]
			savmags = self.trinfo["savmags"]
			savmoms = self.trinfo["savmoms"]
			radlats = self.trinfo["radlats"]
			radlons = self.trinfo["radlons"]
			coslats = self.trinfo["coslats"]
		else:
			savtims = trinfo["savtims"]
			savmags = trinfo["savmags"]
			savmoms = trinfo["savmoms"]
			radlats = trinfo["radlats"]
			radlons = trinfo["radlons"]
			coslats = trinfo["coslats"]


		#setting up empty lists
		momrats = []
		#mag1list = []
		#mag2list = []
		trrats = []
		distbet = []

		#now going through each moment and time, and computing the ratios
		for i in range(0,len(savtims)):
			#calculating ratios and distances
			tmomrats = np.divide(savmoms[i],savmoms[i+1:])
			ttrrats = np.divide(savtims[i],savtims[i+1:])
			dists = self.distcalc(radlats[i],radlons[i],radlats[i+1:],radlons[i+1:],coslats[i],coslats[i+1:])
			dists = np.multiply(dists,1000.) #to m

			#saving to list
			for j in range(len(tmomrats)):
				momrats.append(tmomrats[j])
				trrats.append(ttrrats[j])
				distbet.append(dists[j])
				#mag1list.append(savmags[i])
				#mag2list.append(savmags[i+1+j])


		#plotting histograms of results in distance bins if option is on
		#if ploton is True:
		#	self.momtrratsplot(mag1list,mag2list,distbet)

		#and outputting them
		momrats = np.array(momrats)
		trrats = np.array(trrats)
		distbet = np.array(distbet)
		
		#outputting results
		if returnvals is False:
			self.momrats = momrats
			self.trrats = trrats
			self.distbet = distbet
			
			#getting moment bins in the log domain
			stval = round(np.log10(np.min(momrats))-0.05,1)
			enval = round(np.log10(np.max(momrats))+0.05,1)
			self.momratsbins = np.arange(stval,enval,1.0)
		else:
			return momrats, trrats, distbet
	
	
	#-----------------------------------------------------------------------------------------------------------------------------------#
	#for cases when we want to plot the results for momtrrats to check for errors
	def momtrratsplot(self,mag1list,mag2list,distbet):
		#setting up the magnitude value lists
		mag1list = np.array(mag1list)
		mag2list = np.array(mag2list)
		
		for ij in range(len(dists)-1):
			#finding values in that distance bin
			distlocs = np.argwhere(np.logical_and(distbet>=self.distbins[ij],distbet<self.distbins[ij+1])==True).flatten()

			#taking the magnitudes of those eqs
			tmag1 = mag1list[distlocs]
			tmag2 = mag2list[distlocs]

			#setting up magnitude bins for plotting
			magbins = np.arange(0.5,7.0,0.1)

			#plotting a histogram for this distance bin
			fig20 = plt.figure(20,figsize=(12,8))
			plt.hist(tmag1,bins=magbins,alpha=0.5,label='Mag 1')
			plt.hist(tmag2,bins=magbins,alpha=0.5,label='Mag 2')
			plt.title('Distbin = '+str(self.distbins[ij])+' - '+str(self.distbins[ij+1]))
			plt.xlabel('Magnitudes')
			plt.legend()
			plt.show()	
		
	#-----------------------------------------------------------------------------------------------------------------------------------#
	#plotting the results of momrats and trrats
	#PRETTY SURE THE ERRORPLOT=FALSE OPTION DOES NOT WORK ON THIS PLOT
	def plotrats(self,logplot=False,errorplot=True,zoom=True):
		import pickle
		import io
		#putting distance threshold in metres
		distthresh = self.distthresh*1000.

		#finding events within distance
		close = np.argwhere(self.distbet<distthresh).flatten()

		#setting up figure and plotting
		fig10 = plt.figure(10,figsize=(16,8))

		#calculating values for plotting relation lines
		xvals = np.arange(np.log10(np.min(self.momrats)),np.log10(np.max(self.momrats)),0.1)
		xvals = np.power(10.,xvals)
		thirdvals = np.power(xvals,1./3.)
		sixthvals = np.power(xvals,1./6.)
		twelfthvals = np.power(xvals,1./12.)

		#plotting the points for each earthquake
		#if logplot is False:
		#	plt.scatter(self.momrats[close],self.trrats[close],c=self.distbet[close],marker='.',cmap=plt.get_cmap('autumn'),alpha=0.01)
		#elif logplot is True:
		#	plt.scatter(self.momrats[close],self.trrats[close],c=self.distbet[close],marker='.',cmap=plt.get_cmap('autumn'),norm=mpl.colors.LogNorm(),alpha=0.01)
		
		#plotting the different relations we want to test
		plt.plot(xvals,thirdvals,'g',label='M0 to third')
		plt.plot(xvals,sixthvals,'b',label='M0 to sixth')
		plt.plot(xvals,twelfthvals,'c',label='M0 to twelfth')
		
		#calculating the weighted least squares error on each of the distance bins for each of the
		#three possible relations
		self.weightedls()
		
		
		#get the current axes
		ax = plt.gca()

		#also getting the default color cycle
		col_cycle = [p['color'] for p in plt.rcParams['axes.prop_cycle']]

		
		if errorplot is True:
			#median calculation and plotting goes here
			for jk in range(len(self.distbins)-1):
				self.momdistbins(ax,jk,colour=col_cycle[jk])

			#calculating the median of the entire dataset
			self.momdistbins(ax,0,colour='k',wholecat=True,distbins=np.array([0.0,50000000.0]))

		elif errorplot is False:
			#median calculation and plotting goes here
			for jk in range(len(self.distbins)-1):
				self.momdistbins(ax,jk,errorplt=False,colour=col_cycle[jk])

			#calculating the median of the entire dataset
			self.momdistbins(ax,0,errorplt=False,colour='k',wholecat=True,distbins=np.array([0.0,50000000.0]))


		#rest of plotting
		plt.yscale('log')
		plt.xscale('log')
		#plt.xlim(0.001,500)
		plt.ylim(10**-8,10**8)
		plt.title('DD Parkfield catalogue, eq pairs within '+str(distthresh)+' m only')
		plt.xlabel('Earthquake pair moment ratios (Nm)')
		plt.ylabel('Earthquake pair recurrence time ratios (s)')
		#plt.colorbar(label='Distance between pairs of earthquakes (m)')
		plt.legend(loc='lower right')
		
		#zooming in the figure 
		if zoom is True:
			plt.xlim(10.**-3.,10.**3.)
			plt.ylim(10.**-2.,10.**2.)
		
		#plt.show()
		return fig10

	#-------------------------------------------------------------------------------------------------------------#
	#function to find locations that satisfy the moment and distance bins
	def momdistbins(self,ax,ij,colour=None,errorplt=True,wholecat=False,distbins=None):
		#grabbing the relevant distance bins
		if distbins is None:
			distbins = self.distbins
		
		
		
		#setting percentiles to plot
		lowper = 2.5
		highper = 97.5
		lowper = int(lowper*10.)
		highper = int(highper*10.)
		

		#first finding the distance locations
		distlocs = np.argwhere(np.logical_and(self.distbet>=distbins[ij],self.distbet<distbins[ij+1])==True).flatten()


		#finding the mode/median trrats for the distance/moment bins
		meds = self.modefunc(np.log10(self.momrats[distlocs]),np.log10(self.trrats[distlocs]),self.momratsbins,ploton=False,getmode=True)
		meds = np.power(10,meds)#getting the value out of log

		#plotting results
		if errorplt is False:
			plt.plot(np.power(10,self.momratsbins),meds,'s',color=colour,label='Dists '+str(distbins[ij])+' - '+str(distbins[ij+1])+' m')

		elif errorplt is True:
			for kl in range(len(self.momratsbins)):
				#grabbing the uncertainty out of the bootstrapping results
				#ij defines the distance bin, kl defines the moment bin
				if wholecat is False:
					tempuncerts = np.sort(self.ratiouncert[:,ij,kl])
				elif wholecat is True:
					tempuncerts = np.sort(self.ratiouncert[:,-1,kl])



				#checking the errors aren't nan
				if np.isfinite(tempuncerts[lowper]) == False:
					tempuncerts[lowper] = 10.**10.
				if np.isfinite(tempuncerts[highper]) == False:
					tempuncerts[highper] = 10.**10.

				#making the errors for plotting
				err25 = np.abs(meds[kl]-tempuncerts[lowper])
				err975 = np.abs(meds[kl]-tempuncerts[highper])

				#checking for problems with the whole catalogue error
				#if wholecat is True:
				#	codeint(locals(),globals())

				if err25 == 0.0:
					err25 = 10**-20
				if err975 == 0.0:
					err975 = 10**-20





				#selecting a random number from a generator to shift the x position of the point by 
				#se we can see the medians of all the bins
				shiftnum = np.random.choice(np.arange(-0.1,0.1,0.001))

				#plotting the error bar with x position shifted
				ax.errorbar(np.power(10,self.momratsbins[kl]+shiftnum),meds[kl],yerr=np.array([[err25],[err975]]),fmt='s',color=colour)

		#plotting value to go in the legend
		plt.plot(np.power(10,self.momratsbins[kl]+shiftnum),meds[kl],'s',color=colour,label='Dists '+str(distbins[ij])+' - '+str(distbins[ij+1])+' m')
	
	#-----------------------------------------------------------------------------------------------------------------------------------#
	#calculating the weighted least squares values for each different distance and moment bin
	def weightedls(self):
		#must be run after bootstrapmedians as it requires the values from that function
		#we want to calculate the weighted least squares fit of each distance
		#bin to each of the 1/3 1/6 relations.
		#first need to calculate the relations for the moment bins that I actually use
		thirdrely = np.power(self.momratsbins,1./3.)
		sixthrely = np.power(self.momratsbins,1./6.)
		twelfthrely = np.power(self.momratsbins,1./12.)
		
		#setting up empty array to save the results of wls in for each distance bins and each relation
		wlsarr = np.empty([len(self.distbins),3.])
		
		
		#We loop through each distance bin
		for jk in range(len(self.distbins)-1):
			distlocs = np.argwhere(np.logical_and(self.distbet>=self.distbins[ij],self.distbet<self.distbins[ij+1])==True).flatten()

			#finding the mode/median trrats for the distance/moment bins
			meds = self.modefunc(np.log10(self.momrats[distlocs]),np.log10(self.trrats[distlocs]),self.momratsbins,getmode=True)
			meds = np.power(10,meds)#getting the value out of log
			
			#grabbing the uncertainties for all the moment ratios bins
			tempuncerts = np.sort(self.ratiouncert[:,jk])
			
			#now calculating the weights for each moment bin
			weights = []
			for kl in range(len(self.momratsbins)):
				#grabbing the section of the uncertainties
				reluncert = tempuncerts[kl]
				
				#calculating the variance and thus grabbing the weight
				weights.append(1./np.var(reluncert))
			
			#now calculating the weighted least squares
			thirdwrest = np.sum(np.multiply(weights,np.power(np.subtract(meds,thirdrely),2.)))
			sixthwrest = np.sum(np.multiply(weights,np.power(np.subtract(meds,sixthrely),2.)))
			twelfthwrest = np.sum(np.multiply(weights,np.power(np.subtract(meds,twelfthrely),2.)))
			thirdwls = np.multiply((1./len(self.momratsbins)),thirdwrest)
			sixthwls = np.multiply((1./len(self.momratsbins)),sixthwrest)
			twelfthwls = np.multiply((1./len(self.momratsbins)),twelfthwrest)
			
			#saving the results to an array
			wlsarr[jk,0] = thirdwls
			wlsarr[jk,1] = sixthwls
			wlsarr[jk,2] = twelfthwls
		
		
		#now doing the same thing for the entire catalogue using the maximum distance bin
		distlocs = np.argwhere(np.logical_and(self.distbet>=0.0,self.distbet<5000000.)==True).flatten()

		#finding the mode/median trrats for the distance/moment bins
		meds = self.modefunc(np.log10(self.momrats[distlocs]),np.log10(self.trrats[distlocs]),self.momratsbins,getmode=True)
		meds = np.power(10,meds)#getting the value out of log

		#grabbing the uncertainties for all the moment ratios bins
		tempuncerts = np.sort(self.ratiouncert[:,-1])

		#now calculating the weights for each moment bin
		weights = []
		for kl in range(len(self.momratsbins)):
			#grabbing the section of the uncertainties
			reluncert = tempuncerts[kl]

			#calculating the variance and thus grabbing the weight
			weights.append(1./np.var(reluncert))

		#now calculating the weighted least squares
		thirdwrest = np.sum(np.multiply(weights,np.power(np.subtract(meds,thirdrely),2.)))
		sixthwrest = np.sum(np.multiply(weights,np.power(np.subtract(meds,sixthrely),2.)))
		twelfthwrest = np.sum(np.multiply(weights,np.power(np.subtract(meds,twelfthrely),2.)))
		thirdwls = np.multiply((1./len(self.momratsbins)),thirdwrest)
		sixthwls = np.multiply((1./len(self.momratsbins)),sixthwrest)
		twelfthwls = np.multiply((1./len(self.momratsbins)),twelfthwrest)

		#saving the results to an array
		wlsarr[-1,0] = thirdwls
		wlsarr[-1,1] = sixthwls
		wlsarr[-1,2] = twelfthwls
		
		self.wlsarr = wlsarr
	
	
	#-----------------------------------------------------------------------------------------------------------------------------------#
	#plotting the ruptures versus the times
	def plotruptimes(self):
		
		#extracting
		evrups = self.evinfo["evrups"]
		savrups = self.trinfo["savrups"]
		savtims = self.trinfo["savtims"]

		#calculating ev rupture sizes for use
		savrupslog = np.log10(savrups)
		locs = np.where(np.isfinite(savrups)==True)

		#calculating the rupture size of the completeness magnitude
		evrupcomp = np.divide(np.power(10,np.subtract(2.8,1)),2.)


		#setting up figure
		fig = plt.figure(2,figsize=(16,8))
		gs = gridspec.GridSpec(2,1,height_ratios=[3,1])

		#plotting the tr versus eq rupture size subplot
		ax0 = plt.subplot(gs[0])

		#finding locations where medtims is not nan
		locs2 = np.where(np.isfinite(self.trinfo["medtims"])==True)
		rupbinspow = np.power(10,self.trinfo["rupbins"][locs2])

		#doing rest of plotting
		ax0.plot(savrups,savtims,'b.')
		ax0.plot(rupbinspow,self.trinfo["medtims"][locs2],'ks',markersize=20,label='median')
		ax0.set_xlabel('Earthquake rupture size (m)')
		ax0.set_xscale('log')
		ax0.set_ylabel('Time to the next earthquake within rupture size (s)')
		ax0.set_yscale('log')
		#ax0.set_xlim(0.2,50000)

		#plotting the relation you expect given M0^1/3 and M0^1/6
		#for M0^1/3 you expect tr proportional to r^2/3
		#for M0^1/6 you expect tr proportional to r^1/3
		#for M0^1/12 you expect tr proportional to r^1/6
		tr23 = np.multiply(np.power(rupbinspow,2./3.),np.divide(np.median(self.trinfo["medtims"][locs2]),np.median(np.power(rupbinspow,2./3.))))
		tr13 = np.multiply(np.power(rupbinspow,1./3.),np.divide(np.median(self.trinfo["medtims"][locs2]),np.median(np.power(rupbinspow,1./3.))))
		tr16 = np.multiply(np.power(rupbinspow,1./6.),np.divide(np.median(self.trinfo["medtims"][locs2]),np.median(np.power(rupbinspow,1./6.))))

		#plotting the relations you expect
		ax0.plot(rupbinspow,tr23,'g--',label='M0^1/3')
		ax0.plot(rupbinspow,tr13,'r--',label='M0^1/6')
		ax0.plot(rupbinspow,tr16,'c--',label='M0^1/12')


		#legend
		ax0.legend(loc='lower right')




		#now plotting the histogram of how many values are going into each median
		ax1 = plt.subplot(gs[1])
		bins = np.arange(np.min(savrupslog[locs])-0.05,np.max(savrupslog[locs])+0.15,0.1)
		nums,bns,c = ax1.hist(savrupslog[locs],bins,edgecolor="black",linewidth=1.0)
		ax1.set_xlabel('Rupture size (log10 scale)')
		#ax1.set_xlim(np.log10(0.2),np.log10(50000))

		#plotting line for completeness magnitude
		#ax1.plot([np.log10(evrupcomp),np.log10(evrupcomp)],[0.,np.max(nums)],'k--',linewidth=2.0)

		#calculating the gradient in normal and log space
		gradient, blah = np.polyfit(self.trinfo["rupbins"][locs2],self.trinfo["medtims"][locs2],1)
		gradientlog, blah = np.polyfit(self.trinfo["rupbins"][locs2],np.log10(self.trinfo["medtims"][locs2]),1)
		print("Linear gradient: "+str(gradient))
		print("Log-Linear gradient: "+str(gradientlog))

		#plt.tight_layout()
		#plt.show()
		return fig

	#-----------------------------------------------------------------------------------------------------------------------------------#
	#plotting earthquake moments versus the times
	def plotmomtimes(self):
		
		#extracting
		evrups = self.evinfo["evrups"]
		savrups = self.trinfo["savrups"]
		savmoms = self.trinfo["savmoms"]
		savtims = self.trinfo["savtims"]
		savmags = self.trinfo["savmags"]
		

		#taking log of moments to make it easier to bin
		savmomslog = np.log10(savmoms)
		locs = np.where(np.isfinite(savmoms)==True)

		#setting up figure
		fig = plt.figure(1,figsize=(16,8))
		gs = gridspec.GridSpec(2,1,height_ratios=[3,1])

		#plotting the tr versus eq rupture size subplot
		ax0 = plt.subplot(gs[0])

		#binning the event moments
		stval = round(np.min(savmomslog)-0.05,1)
		enval = round(np.max(savmomslog)+0.05,1)
		mombins = np.arange(stval,enval,0.5)

		#getting the mode of each bin
		medtims = self.modefunc(savmomslog,savtims,mombins)

		#finding locations where medtims is not nan
		locs2 = np.where(np.isfinite(medtims)==True)[0]
		mombinspow = np.power(10,mombins[locs2])


		#doing rest of plotting
		ax0.plot(savmoms,savtims,'b.',alpha=0.2)

		#plotting each median with the appropriate uncertainty
		#looping through each moment bin and plotting the median and error estimate

		for kl in range(len(locs2)):
			#grabbing the uncertainty out of the bootstrapping results
			tempuncerts = np.sort(self.pairuncert[:,locs2[kl]])

			#checking the errors aren't nan
			if np.isfinite(tempuncerts[25]) == False:
				tempuncerts[25] = 10.**10.
			if np.isfinite(tempuncerts[975]) == False:
				tempuncerts[975] = 10.**10.

			#making the errors for plotting
			err25 = np.abs(medtims[locs2[kl]]-tempuncerts[25])
			err975 = np.abs(medtims[locs2[kl]]-tempuncerts[975])

			if err25 == 0.0:
				err25 = 10**-20
			if err975 == 0.0:
				err975 = 10**-20


			#plotting the error bar
			ax0.errorbar(mombinspow[kl],medtims[locs2[kl]],yerr=np.array([[err25],[err975]]),fmt='ks',markersize=15)

		#setting up median plot for use in the legend
		ax0.plot(mombinspow[kl],medtims[locs2[kl]],'ks',markersize=20,label='median',alpha=0.0)

		#doing some other plot formatting stuff
		ax0.set_xlabel('Earthquake moment (N m)')
		ax0.set_xscale('log')
		ax0.set_ylabel('Time to the next earthquake within rupture size (s)')
		ax0.set_yscale('log')
		#ax0.set_xlim(0.2,50000)

		#plotting the relation you expect given M0^1/3 and M0^1/6
		#for M0^1/3 you expect tr proportional to r^2/3
		#for M0^1/6 you expect tr proportional to r^1/3
		#for M0^1/12 you expect tr proportional to r^1/6
		#tr13 = np.multiply(np.power(mombinspow,1./3.),np.median(medtims[locs2]))
		#tr16 = np.multiply(np.power(mombinspow,1./6.),np.median(medtims[locs2]))
		tr13 = np.multiply(np.power(mombinspow,1./3.),np.divide(np.median(medtims[locs2]),np.median(np.power(mombinspow,1./3.))))
		tr16 = np.multiply(np.power(mombinspow,1./6.),np.divide(np.median(medtims[locs2]),np.median(np.power(mombinspow,1./6.))))
		tr112 = np.multiply(np.power(mombinspow,1./12.),np.divide(np.median(medtims[locs2]),np.median(np.power(mombinspow,1./12.))))

		#tr16test = np.multiply(np.power(mombinspow,1./6.),1000000)

		#plotting the relations you expect
		ax0.plot(mombinspow,tr13,'g--',label='M0^1/3')
		ax0.plot(mombinspow,tr16,'r--',label='M0^1/6')
		ax0.plot(mombinspow,tr112,'c--',label='M0^1/12')
		
		#legend
		ax0.legend(loc='lower right')

		#now plotting the histogram of how many values are going into each median
		ax1 = plt.subplot(gs[1])
		bins = np.arange(np.min(savmomslog[locs])-0.05,np.max(savmomslog[locs])+0.15,0.1)
		nums,bns,c = ax1.hist(savmomslog[locs],bins,edgecolor="black",linewidth=1.0)
		ax1.set_xlabel('Earthquake moment (log10 scale)')
		#ax1.set_xlim(np.log10(0.2),np.log10(50000))

		#calculating the gradient in normal and log space
		try:
			gradient, blah = np.polyfit(mombins[locs2],medtims[locs2],1)
			gradientlog, blah = np.polyfit(mombins[locs2],np.log10(medtims[locs2]),1)
			print("Linear gradient: "+str(gradient))
			print("Log-Linear gradient: "+str(gradientlog))
		except:
			print('Gradient failed')

		#plt.tight_layout()
		#plt.show()
		return fig
	#------------------------------------------------------------------------------------------------------------------------------------#
	#for grabbing the mode of a dataset
	def modefunc(self,var1,var2,var1bins,getmode=True,ploton=False):
		#THIS CODE ASSUMES LOG BINS AND LOG VAR1
		#getting mode results
		moderes = []

		#first identifying the bin width and half bin width for checking if the values are within the bins
		binwidth = np.abs(np.subtract(var1bins[1],var1bins[0]))
		halfwidth = 0.5*binwidth
		for val in range(len(var1bins)):


			#finding values that sit in the bins
			bindiff = np.subtract(var1,var1bins[val])
			dlocs = np.argwhere(np.logical_and(bindiff<=halfwidth,bindiff>-(halfwidth-0.000000000000000000000000001))).flatten()

			#get the histogram of the data set so we can get the mode
			tn,tbins = np.histogram(var2[dlocs],bins=40)

			#find the maximum location of the bins
			tloc = np.argwhere(tn==np.max(tn)).flatten()

			#testing for where the mode is over several data points
			if len(tloc) > 1:
				#just take the median over the bins to pick a middle value
				tmode = np.median([tbins[tloc[0]],tbins[tloc[-1]+1]])


			#testing for when there are no data points
			if tn[tloc[0]] != 0:
				#now finding the mode of the histogram
				tmode = np.median([tbins[tloc],tbins[tloc+1]])
			else:
				tmode = np.nan

			#and if there is only one data point then take that value
			if len(dlocs) == 1:
				tmode = var2[dlocs[0]]

			#plotting for testing
			#if ploton is True:	
			#	plt.figure(100)
			#	plt.hist(var2[dlocs],bins=40)
			#	plt.plot([np.median(var2[dlocs]),np.median(var2[dlocs])],[0.,10],label='median')
			#	plt.plot([tmode,tmode],[0.,10],label='mode')
			#	plt.legend(loc='upper right')
			#	plt.show()


			#either taking the mode of the median
			if getmode is True:
				moderes.append(tmode)
			elif getmode is False:
				moderes.append(np.median(var2[dlocs]))

		#returning array of modes for bins
		moderes = np.array(moderes)
		return moderes

	#------------------------------------------------------------------------------------------------------------------------------------#
	#estimating the uncertainties on medians for time time2next eq and ratios plots
	def bootstrapmedians(self,numruns=1000):
		#setting up original list of numbers for events
		evtcs = np.arange(0,len(self.evinfo["evdates"])-1,1)

		#getting the moment bins so I can keep them consistent over all the bootstrapped uncertainties
		savmoms = self.trinfo["savmoms"]
		savmomslog = np.log10(savmoms)
		stval = round(np.min(savmomslog)-0.05,1)
		enval = round(np.max(savmomslog)+0.05,1)
		mombins = np.arange(stval,enval,0.5)

		#moment and distance bins for the ratios
		#getting moment bins in the log domain
		momratsbins = self.momratsbins
		

		#setting number of events to select in each bootstrapping instance
		nevts = int(len(evtcs)*0.8)

		#looping through the number of times to estimate
		pairuncert = np.empty([numruns,len(mombins)])
		ratiouncert = np.empty([numruns,len(self.distbins),len(momratsbins)])
		templist = []
		for i in range(numruns):
			print('Bootstrap run '+str(i+1)+' of '+str(numruns))
			#selecting a new set of events
			newevtc = np.sort(np.random.choice(evtcs,nevts,replace=False))


			#recreating evinfo but only with these new events
			evinfo2 = {"evid":self.evinfo["evid"][newevtc],"evdates":self.evinfo["evdates"][newevtc],"evlats":self.evinfo["evlats"][newevtc],
					"evlons":self.evinfo["evlons"][newevtc],"evdeps":self.evinfo["evdeps"][newevtc],"evmags":self.evinfo["evmags"][newevtc],
					"evmoms":self.evinfo["evmoms"][newevtc],"evmagtype":self.evinfo["evmagtype"][newevtc],"evrups":self.evinfo["evrups"][newevtc],
					"radlats":self.evinfo["radlats"][newevtc],"radlons":self.evinfo["radlons"][newevtc],"coslats":self.evinfo["coslats"][newevtc]}

			#now redoing the time to next earthquake and ratios calculations
			trinfo2 = self.calctr(evinfo=evinfo2,returnvals=True)
			momrats2, trrats2, distbet2 = self.momtrrats(trinfo=trinfo2,returnvals=True)

			#calculating the medians/modes for the moment bins using the newly sampled catalogue
			medtims2 = self.modefunc(np.log10(trinfo2["savmoms"]),trinfo2["savtims"],mombins)

			#setting up empty array to save the results for the medians
			temparr = np.empty([len(self.distbins),len(momratsbins)])

			#now calculating the medians/modes for the ratios in moment bins
			for ij in range(len(self.distbins)-1):
				#finding distance locations
				distlocs = np.argwhere(np.logical_and(distbet2>=self.distbins[ij],distbet2<self.distbins[ij+1])==True).flatten()

				#getting the mode/median
				meds = self.modefunc(np.log10(momrats2[distlocs]),np.log10(trrats2[distlocs]),momratsbins)
				meds = np.power(10,meds)#getting the value out of log
				temparr[ij] = meds

			#running the same process for a distance bin including the entire dataset
			distlocs = np.argwhere(np.logical_and(distbet2>=0.0,distbet2<50000000.0)==True).flatten()
			medswholecat = self.modefunc(np.log10(momrats2[distlocs]),np.log10(trrats2[distlocs]),momratsbins)
			medswholecat = np.power(10,medswholecat)
			temparr[-1] = medswholecat

			#saving the final results
			pairuncert[i] = medtims2
			ratiouncert[i] = temparr


		self.pairuncert = pairuncert
		self.ratiouncert = ratiouncert


	#------------------------------------------------------------------------------------------------------------------------------------#
	#plotting histogram of rupture lengths
	def pltruphist(self):
		fig = plt.figure(2,figsize=(16,8))
		#getting rupture lengths (radius)
		evrups = self.evinfo["evrups"]
		evrupslog = np.log10(evrups)


		#locate nans
		locs = np.where(np.isfinite(evrups)==True)

		#making bins for histogram
		bins = np.arange(np.min(evrupslog[locs])-0.05,np.max(evrupslog[locs])+0.05,0.1)

		#plotting
		plt.hist(evrupslog[locs],bins,edgecolor="black",linewidth=1.0)
		plt.xlabel('Rupture size (log10 scale)')
		plt.show()

