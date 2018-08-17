import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import regionprops, label
from matplotlib.widgets import Slider, Button, CheckButtons, Cursor
from lungmask_pro import lungmask_pro, lungmask3D
from scipy.ndimage import sobel
from image_functions import maxminscale
from mayavi import mlab
from scipy.ndimage.interpolation import zoom
from pytictoc import TicToc
from grow_in_3d import level_set3D
import time
from unet_VGG import pred_VGG


# input_im:		ct image
# output_im:	ground truth image
# pred_output:	predicted output
# opinion_func:	function that retuns different radiologists opinions
# thr: 			threshold

def plot2d(input_im, output_im, pred_output, opinion_func, thr = 0.5):
	pred_tmp = pred_output.copy()
	pred_tmp[pred_tmp < thr] = 0 

	opinions = opinion_func

	tmp = np.zeros((input_im.shape[0], opinions.shape[0], input_im.shape[1], input_im.shape[2]))
	opin = np.zeros((input_im.shape[0], input_im.shape[1], input_im.shape[2]))
	for i in range(input_im.shape[0]):
		for doc in range(opinions.shape[0]):
			tmp[i, doc] = cv2.resize(opinions[doc, i, :, :, 0],(input_im.shape[1],input_im.shape[2])) 

		opin[i] = np.add( np.add(tmp[i, 0, :, :], tmp[i, 1, :, :]),
						np.add(tmp[i, 2, :, :], tmp[i, 3, :, :]))

	for i in range(input_im.shape[0]):
		if (np.count_nonzero(pred_tmp[i, :, :, 1]) == 0) and (np.count_nonzero(output_im[i, :, :, 1]) == 0):
			continue 
		

		f, ax = plt.subplots(2,3, figsize = (20, 12))
		f.suptitle('slice '+str(i))

		ax[0,0].imshow(output_im[i, :, :, 1], cmap= 'gray')
		ax[0,0].set_title('output')
		ax[0,0].set_axis_off()

		ax[0,1].imshow(pred_output[i, :, :, 1], cmap = 'gray', vmin = 0, vmax = 1)
		ax[0,1].set_title('predicted')
		ax[0,1].set_axis_off()

		ax[0,2].imshow(input_im[i, :, :, 0], cmap = 'gray')
		ax[0,2].set_title('ct image')
		ax[0,2].set_axis_off()

		ax[1,1].imshow(input_im[i, :, :, 0], cmap = 'gray')
		ax[1,1].imshow(pred_output[i, :, :, 1], cmap = 'inferno', alpha = 0.35, vmin = 0, vmax = 1)
		ax[1,1].set_title('predicted on ct')
		ax[1,1].set_axis_off()
		
		ax[1,0].imshow(input_im[i, :, :, 0], cmap = 'gray')
		ax[1,0].imshow(output_im[i, :, :, 1], cmap= 'inferno', alpha = 0.35)
		ax[1,0].set_title('output on ct')
		ax[1,0].set_axis_off()

		ax[1,2].imshow(input_im[i, :, :, 0], cmap = 'gray')
		ax[1,2].imshow(opin[i], cmap = 'inferno', alpha = 0.35, vmin = 0, vmax = 255)
		ax[1,2].set_title("doctor's opinions")
		ax[1,2].set_axis_off()

		plt.show()


# input_im:		ct image
# output_im:	ground truth image
# pred_output:	predicted output
# opinion_func:	function that retuns different radiologists opinions
def slider(input_im, output_im, pred_output, opinion_func):
	#--- slider functions ----
	def new_thr(event):
		tmp = pred_output.copy()
		threshold = event
		tmp[tmp < threshold] = 0
		tmp[tmp >= threshold] = 1

		ax[0,1].imshow(input_im[int(slider2.val), :, :, 0], cmap = 'gray')
		ax[0,1].imshow(tmp[int(slider2.val), :, :, 1], cmap = 'inferno', vmin = 0, vmax = 1, alpha = 0.3)
		ax[0,1].set_title('predicted')
		ax[0,1].set_axis_off()

		f.suptitle('slice '+str(slider2.val))
		f.canvas.draw_idle()

	def images(event):
		tmp = pred_output.copy()
		threshold = slider1.val
		tmp[tmp < threshold] = 0
		tmp[tmp >= threshold] = 1

		ax[0,0].imshow(input_im[int(slider2.val), :, :, 0], cmap = 'gray')
		ax[0,0].imshow(output_im[int(slider2.val), :, :, 1], cmap= 'inferno', alpha = 0.3)
		ax[0,0].set_title('output')
		ax[0,0].set_axis_off()

		ax[0,1].imshow(input_im[int(slider2.val), :, :, 0], cmap = 'gray')
		ax[0,1].imshow(tmp[int(slider2.val), :, :, 1], cmap = 'inferno', vmin = 0, vmax = 1, alpha = 0.3)
		ax[0,1].set_title('predicted')
		ax[0,1].set_axis_off()

		ax[1,0].imshow(input_im[int(slider2.val), :, :, 0], cmap = 'gray')
		ax[1,0].set_title('ct image')
		ax[1,0].set_axis_off()

		ax[1,1].imshow(input_im[int(slider2.val), :, :, 0], cmap = 'gray')
		ax[1,1].imshow(opin[int(slider2.val)], cmap = 'inferno', alpha = 0.35, vmin = 0, vmax = 255)
		ax[1,1].set_title("doctor's opinions")
		ax[1,1].set_axis_off()
		f.suptitle('slice '+str(slider2.val))
		f.canvas.draw_idle()

	def next_pat(event):
		plt.close()

	def next_im(event):
		if slider2.val < input_im.shape[0]:
			slider2.set_val(slider2.val + 1)

	def prev_im(event):
		if slider2.val != 0:
			slider2.set_val(slider2.val - 1)

	def quit(event):
		plt.close()
		exit()
	# ---------------

	opinions = opinion_func
	tmp = np.zeros((input_im.shape[0], opinions.shape[0], input_im.shape[1], input_im.shape[2]))
	opin = np.zeros((input_im.shape[0], input_im.shape[1], input_im.shape[2]))
	for i in range(tmp.shape[0]):
		for doc in range(opinions.shape[0]):
			tmp[i, doc] = cv2.resize(opinions[doc, i, :, :, 0],(input_im.shape[1],input_im.shape[2])) 

		opin[i] = np.add( np.add(tmp[i, 0, :, :], tmp[i, 1, :, :]),
						np.add(tmp[i, 2, :, :], tmp[i, 3, :, :]))

	f, ax = plt.subplots(2,2, figsize = (12, 12))
	b1ax = plt.axes([0.05, 0.10, 0.07, 0.03])
	b2ax = plt.axes([0.05, 0.05, 0.07, 0.03])
	b3ax = plt.axes([0.80, 0.10, 0.07, 0.03])
	b4ax = plt.axes([0.80, 0.05, 0.07, 0.03])

	s1ax = plt.axes([0.25, 0.08, 0.5, 0.03])
	s2ax = plt.axes([0.25, 0.02, 0.5, 0.03])

	slider1 = Slider(s1ax, 'threshold', 0.05, 1.0, dragging = True, valstep = 0.05)
	slider2 = Slider(s2ax, 'slice', 0.0, input_im.shape[0], valstep = 1)
	button1 = Button(b1ax, 'exit', color = 'beige', hovercolor = 'beige')
	button2 = Button(b2ax, 'next patient', color = 'beige',hovercolor = 'beige')
	button3 = Button(b3ax, '-->', color = 'beige', hovercolor = 'beige')
	button4 = Button(b4ax, '<--', color = 'beige',hovercolor = 'beige')

	slider1.set_val(0.5)
	slider2.set_val(0)
	f.subplots_adjust(bottom = 0.15)

	ax[0,0].imshow(input_im[int(slider2.val), :, :, 0], cmap = 'gray')
	ax[0,0].imshow(output_im[int(slider2.val), :, :, 1], cmap= 'inferno', alpha = 0.3)
	ax[0,0].set_title('output')
	ax[0,0].set_axis_off()

	ax[0,1].imshow(input_im[int(slider2.val), :, :, 0], cmap = 'gray')
	ax[0,1].imshow(pred_output[int(slider2.val), :, :, 1], cmap = 'inferno', vmin = 0, vmax = 1, alpha = 0.3)
	ax[0,1].set_title('predicted')
	ax[0,1].set_axis_off()

	ax[1,0].imshow(input_im[int(slider2.val), :, :, 0], cmap = 'gray')
	ax[1,0].set_title('ct image')
	ax[1,0].set_axis_off()

	ax[1,1].imshow(input_im[int(slider2.val), :, :, 0], cmap = 'gray')
	ax[1,1].imshow(opin[int(slider2.val)], cmap = 'inferno', alpha = 0.35, vmin = 0, vmax = 255)
	ax[1,1].set_title("doctor's opinions")
	ax[1,1].set_axis_off()

	f.suptitle('slice '+str(slider2.val))

	button1.on_clicked(quit)
	button2.on_clicked(next_pat)
	button3.on_clicked(next_im)
	button4.on_clicked(prev_im)

	slider1.on_changed(new_thr)
	slider2.on_changed(images)


	plt.show()

	


# im:			the image stack to plot in 3d, in the shape [slices,x,y,2]
# plot_shape:	the tuple in x-y shape of the 3d plot
# title:		the title of the plot
# fig_nr:		figure nr
# thr:			threshold to make the binary plot

# need to call plt.show() after this function
def plot3d(im_st, plot_shape = (64,64), title = 'title',fig_nr = 1, thr = 0.5):
	tmp = im_st.copy()
	tmp[tmp < thr] = 0 
	tmp[tmp >= thr] = 1
	
	rescaled_im = np.zeros((tmp.shape[0],plot_shape[0],plot_shape[1]))
	for i in range(im_st.shape[0]):
		rescaled_im[i, :, :] = cv2.resize(np.uint8(tmp[i, :, :,1]), (plot_shape[0], plot_shape[1]), interpolation=cv2.INTER_LINEAR)
	fig = plt.figure(fig_nr, figsize = (10,10))
	ax = fig.gca(projection='3d')
	ax.set_title(title)
	ax.voxels(rescaled_im)
	ax.set_zlabel('x')
	ax.set_ylabel('y')
	ax.set_xlabel('z')



def plot_lungmask(im_st):
	mask = np.zeros((im_st.shape[0], im_st.shape[1], im_st.shape[2], 2))
	print('get mask')
	#for i in range(im_st.shape[0]):
	#	mask[i, :, :, 1] = lungmask(im_st[i, :, :, 0])
	mask[:,:,:,1] = lungmask3D(im_st[:, :, :, 0], morph = False)
	for i in range(im_st.shape[1]):
		mask[:, i, :, 1] = cv2.Canny(np.uint8(maxminscale(mask[:, i, :, 1])),50,100)

	print('plotting')

	mlab.contour3d(mask[:, :, :, 1], colormap = 'Blues', opacity = 0.1)
	mlab.show()




def plot_lung_and_tumor(im_st, pred_st, gt_st, res = [], thr = 0.5):
	mask = np.zeros((im_st.shape[0],im_st.shape[1], im_st.shape[2],2))

	#for i in range(im_st.shape[0]):
	#	mask[i, :, :, 1] = lungmask_pro(im_st[i, :, :, 0], morph = False)
	mask[:,:,:,1] = lungmask3D(im_st[:, :, :, 0], morph = False)

	for i in range(im_st.shape[1]):
		mask[:, i, :, 1] = cv2.Canny(np.uint8(maxminscale(mask[:, i, :, 1])),50,100)
	
	pred_st[pred_st < thr] = 0 
	pred_st[pred_st >= thr] = 180
	gt_st[gt_st >= thr] = 180	

	# TODO - make zoom_fac dependent on resolution in z dir (true values)
	if not res:
		zoom_fac = im_st.shape[1]/im_st.shape[0]
	else: 
		zoom_fac = res[0]/res[1]/2

	
	mask = zoom(mask[:,:,:,1], zoom = [zoom_fac,1,1])
	pred_st = zoom(pred_st[:, :, :, 1], zoom = [zoom_fac,1,1])
	gt_st = zoom(gt_st[:, :, :, 1], zoom = [zoom_fac,1,1])

	mlab.figure(1, size = (1000,800))
	mlab.contour3d(pred_st, colormap = 'hot', opacity = 1.0, vmin = 0, vmax = 255)
	mlab.contour3d(mask, colormap = 'Greys', opacity = 0.1, vmin = 0, vmax = 150)
	mlab.title('Prediction')

	mlab.figure(2, size = (1000,800))
	mlab.contour3d(gt_st, colormap = 'hot', opacity = 1.0, vmin = 0, vmax = 255)
	mlab.contour3d(mask, colormap = 'Greys', opacity = 0.1, vmin = 0, vmax = 150)
	mlab.title('Ground truth')
	
	mlab.show()

def plot_3d_no_gt(im_st, pred_st, res, thr = 0.5, plot_size_xy = 256):

	zoom_fac = [res[0]/res[1]/(im_st.shape[1]/plot_size_xy), plot_size_xy/im_st.shape[1], plot_size_xy/im_st.shape[1]]
	im_st = zoom(im_st.copy(), zoom = zoom_fac, order = 1)

	mask = np.zeros((im_st.shape[0],im_st.shape[1], im_st.shape[2]))

	#for i in range(im_st.shape[0]):
	#	mask[i, :, :] = lungmask_pro(im_st[i, :, :], morph = False)
	mask[:,:,:] = lungmask3D(im_st[:, :, :], morph = False)

	for i in range(im_st.shape[1]):
		mask[:, i, :] = cv2.Canny(np.uint8(maxminscale(mask[:, i, :])),50,100)
	
	pred_st[pred_st < thr] = 0 
	pred_st[pred_st >= thr] = 180

	pred_st = zoom(pred_st.copy(), zoom = zoom_fac, order = 0)

	
	mlab.figure(1, size = (1000,800))
	mlab.contour3d(pred_st, colormap = 'hot', opacity = 1.0, vmin = 0, vmax = 255)
	mlab.contour3d(mask, colormap = 'Greys', opacity = 0.1, vmin = 0, vmax = 150)
	mlab.title('Prediction')

	mlab.show()



def test2d3d(im_st, pred_st, res, thr = 0.1, plot_size_xy = 150, VGG_model_path = ''):
	
	t = TicToc()
	im_ct = im_st.copy()
	zoom_fac = [res[0]/res[1]/(im_st.shape[1]/plot_size_xy), plot_size_xy/im_st.shape[1], plot_size_xy/im_st.shape[1]]

	im_st = zoom(im_st.copy(), zoom = zoom_fac, order = 1)
	mask = np.zeros((im_st.shape[0],im_st.shape[1], im_st.shape[2]))

	#for i in range(im_st.shape[0]):
	#	mask[i, :, :] = lungmask_pro(im_st[i, :, :], morph = False)
	mask[:,:,:] = lungmask3D(im_st[:, :, :], morph = False)

	for i in range(im_st.shape[1]):
		mask[:, i, :] = cv2.Canny(np.uint8(maxminscale(mask[:, i, :])),50,100)
	
	pred_st_tmp = pred_st.copy()

	pred_st_tmp = zoom(pred_st, zoom = zoom_fac, order = 0)
	bin_pred_3d = pred_st_tmp.copy()
	bin_pred_3d[bin_pred_3d < thr] = 0
	bin_pred_3d[bin_pred_3d >= thr] = 1

	bin_pred_2d = pred_st.copy()
	bin_pred_2d[bin_pred_2d < thr] = 0
	bin_pred_2d[bin_pred_2d >= thr] = 1


	#labels 
	label_3d = label(bin_pred_3d)
	label_2d = label(bin_pred_2d)
	print('Number of nodules found: ' + str(len(np.unique(label_2d))-1))

	mlab.figure(1, size = (1000,800))
	mlab.contour3d(pred_st_tmp, colormap = 'hot', opacity = 1.0, vmin = 0, vmax = 1)

	
	def new_thr(event):
		 
		tmp_thr = pred_st[int(slider2.val)].copy()
		tmp_3d = pred_st_tmp.copy()
		threshold = event

		tmp_thr[tmp_thr < threshold] = 0
		tmp_thr[tmp_thr >= threshold] = 1

		tmp_3d[tmp_3d < threshold] = 0
		tmp_3d[tmp_3d >= threshold] = 1

		mlab.clf()
		#mlab.figure(1, size = (1000,800))
		mlab.contour3d(tmp_3d, colormap = 'hot', opacity = 1.0, vmin = 0, vmax = 1)
		if button1.get_status()[0]:
			mlab.contour3d(mask, colormap = 'Greys', opacity = 0.1, vmin = 0, vmax = 150)
		#mlab.title('Prediction')

		ax.clear()
		ax.imshow(im_ct[int(slider2.val), :, :], cmap = 'gray')
		#ax.imshow(tmp_thr, cmap = 'inferno', vmin = 0, vmax = 1, alpha = 0.3)
		ax.imshow(tmp_thr, cmap = 'gnuplot', vmin = 0, vmax = 2, alpha = 0.3)
		#ax.set_title('predicted')
		ax.set_axis_off()

		f.suptitle('slice '+str(slider2.val))
		f.canvas.draw_idle()


	def images(event):
		
		threshold = slider1.val
		tmp = pred_st[int(event)].copy()
		
		tmp[tmp < threshold] = 0
		tmp[tmp >= threshold] = 1

		ax.clear()
		ax.imshow(im_ct[int(event), :, :], cmap = 'gray')
		#ax.imshow(tmp, cmap = 'inferno', vmin = 0, vmax = 1, alpha = 0.3)
		ax.imshow(tmp, cmap = 'gnuplot', vmin = 0, vmax = 2, alpha = 0.3)
		ax.set_title('predicted')
		ax.set_axis_off()

		
		f.suptitle('slice '+str(int(slider2.val)))
		f.canvas.draw_idle()

	def up_scroll_alt(event):
		if event.key == "up":
			if (slider2.val + 2 > im_ct.shape[0]):
				1
				#print("Whoops, end of stack", print(slider2.val))
			else:
				slider2.set_val(slider2.val + 1)
		

	def down_scroll_alt(event):
		if event.key == "down":
			if (slider2.val - 1 < 0):
				1
				#print("Whoops, end of stack", print(slider2.val))
			else:
				slider2.set_val(slider2.val - 1)


	def up_scroll(event):
		if event.button == 'up':
			if (slider2.val + 2 > im_ct.shape[0]):
				1
				#print("Whoops, end of stack", print(slider2.val))
			else:
				slider2.set_val(slider2.val + 1)


	def down_scroll(event):
		if event.button == 'down':
			if (slider2.val - 1 < 0):
				1
				#print("Whoops, end of stack", print(slider2.val))
			else:
				slider2.set_val(slider2.val - 1)

		

	def show_lung(event):
		if not button4.get_status()[0]:
			tmp_3d = pred_st_tmp.copy()
			threshold = slider1.val

			tmp_3d[tmp_3d < threshold] = 0
			tmp_3d[tmp_3d >= threshold] = 1

			mlab.clf()
			mlab.contour3d(tmp_3d, colormap = 'hot', opacity = 1.0, vmin = 0, vmax = 1)
			if button1.get_status()[0]:
				mlab.contour3d(mask, colormap = 'Greys', opacity = 0.1, vmin = 0, vmax = 150)

		elif button4.get_status()[0]:
			pred_class = pred_VGG(im_ct, pred_st, res, VGG_model_path)	
			pred_class = zoom(pred_class.copy(), zoom = zoom_fac, order = 0)
			mlab.clf()
			print(np.unique(pred_class)[1:])

			for i in np.unique(pred_class):
				if i == 0:
					continue
				tmp = pred_class.copy()
				tmp[pred_class != i] = 0
				mlab.contour3d(tmp, colormap = 'OrRd', color = tuple(colors[int(round((9/5)*i)),:]), vmin = 1, vmax = 5)
				mlab.scalarbar(orientation = 'vertical', nb_labels = 9, label_fmt='%.1f')

			if button1.get_status()[0]:
				mlab.contour3d(mask, colormap = 'Greys', opacity = 0.1, vmin = 0, vmax = 150)

		mlab.orientation_axes(xlabel = 'z', ylabel = 'y', zlabel = 'x')


	def remove(event):
		# only add when seed point selected is at an axis
		if (event.ydata != None) or (event.xdata != None):
			ix, iy = int(event.ydata), int(event.xdata)

			# if already in add mode -> switch to remove mode
			# if button3.get_status()[0]:
			# 	button3.set_active(0)

		#print(event)

		if (str(event.inaxes).split('(')[0] == 'AxesSubplot') and button2.get_status()[0]:
			# remove
			coords_zoom = (np.array([slider2.val, ix, iy])*np.array(zoom_fac)).astype(int)
			coords_orig = (int(slider2.val), int(ix), int(iy))
			print(coords_zoom,coords_orig)

			val_3d = label_3d[tuple(coords_zoom)]
			pred_st_tmp[label_3d == val_3d] = 0

			val_2d = label_2d[coords_orig]
			pred_st[label_2d == val_2d] = 0

			# re-plot
			tmp_thr = pred_st[int(slider2.val)].copy()
			tmp_3d = pred_st_tmp.copy()
			threshold = slider1.val

			tmp_thr[tmp_thr < threshold] = 0
			tmp_thr[tmp_thr >= threshold] = 1

			tmp_3d[tmp_3d < threshold] = 0
			tmp_3d[tmp_3d >= threshold] = 1

			mlab.clf()
			mlab.contour3d(tmp_3d, colormap = 'hot', opacity = 1.0, vmin = 0, vmax = 1)
			if button1.get_status()[0]:
				mlab.contour3d(mask, colormap = 'Greys', opacity = 0.1, vmin = 0, vmax = 150)

			ax.clear()
			ax.imshow(im_ct[int(slider2.val), :, :], cmap = 'gray')
			#ax.imshow(tmp_thr, cmap = 'inferno', vmin = 0, vmax = 1, alpha = 0.3)
			ax.imshow(tmp_thr, cmap = 'gnuplot', vmin = 0, vmax = 2, alpha = 0.3)
			cursor = Cursor(ax, useblit=True, color='orange', linewidth=0.5)
			#ax.set_title('predicted')
			ax.set_axis_off()

			f.suptitle('slice '+str(slider2.val))
			f.canvas.draw_idle()



	# def add_window(event):
	# 	# only add when seed point selected is at an axis
	# 	if (event.ydata != None) or (event.xdata != None):
	# 		ix, iy = int(event.ydata), int(event.xdata)

	# 	if (str(event.inaxes).split('(')[0] == 'AxesSubplot') and button3.get_status()[0]:
	# 		figs.show()

	def add(event):

		def adv_exit(event):
			print(1)
			plt.close(figs)
			#plt.close()

		def adv_start(event):
			print(2)
			plt.close(figs)

		def adv_window():
			figs, axx = plt.subplots(num = 'Advanced settings')
			figs.canvas.mpl_connect('key_press_event', adv_exit)
			axx.axis('off')
			#bx1_as = plt.axes([0.05, 0.3, 0.15, 0.11])
			#bx1_as.set_axis_off()

			#button_as1 = CheckButtons(bx1_as, ['lambda1'], [1])
			axx.axis('off')
			ax_as1 = plt.axes([0.15, 0.02, 0.5, 0.05])
			slider_as1 = Slider(ax_as1, 'lambda1', 0.1, 4, dragging = True, valstep = 0.1)

			ax_as2 = plt.axes([0.15, 0.10, 0.5, 0.05])
			slider_as2 = Slider(ax_as2, 'lambda2', 0.1, 4, dragging = True, valstep = 0.1)

			ax_as3 = plt.axes([0.15, 0.18, 0.5, 0.05])
			slider_as3 = Slider(ax_as3, 'smoothing', 0, 4, dragging = True, valstep = 1)

			ax_as4 = plt.axes([0.15, 0.26, 0.5, 0.05])
			slider_as4 = Slider(ax_as4, 'iterations', 1, 1000, dragging = True, valstep = 1)

			ax_as5 = plt.axes([0.15, 0.34, 0.5, 0.05])
			slider_as5 = Slider(ax_as5, 'radius', 0.5, 5, dragging = True, valstep = 0.1)

			ax_b1 = plt.axes([0.85, 0.15, 0.07, 0.08])
			ax_b2 = plt.axes([0.85, 0.05, 0.07, 0.08])

			but_as1 = Button(ax_b1, 'exit', color = 'beige', hovercolor = 'beige')
			but_as2 = Button(ax_b2, 'start', color = 'beige', hovercolor = 'beige')

			#ax_textbox = plt.axes([0, 0.4, 0.5, 0.4])
			#axx.axis('off')
			textstr = "Press ENTER in terminal to start segmentation. \n Shouldn't be necessairy to change settings below, but can be tuned if \n resulting segmentation is not ideal. \n Especially if small nodule: try setting lambda1 <= lambda2 \n Or if very nonhomogeneous nodule: try setting lambda1 > lambda2."

			props = dict(boxstyle='round', facecolor='wheat')
			axx.text(-0.18, 0.25, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)


			slider_as1.set_val(lambda1)
			slider_as2.set_val(lambda2)
			slider_as3.set_val(smoothing)
			slider_as4.set_val(iterations)
			slider_as5.set_val(rad)
			but_as1.on_clicked(adv_exit)
			but_as2.on_clicked(adv_start)

			figs.canvas.draw_idle()

			return figs, axx, slider_as1, slider_as2, slider_as3, slider_as4, slider_as5


		# only add when seed point selected is at an axis
		if (event.ydata != None) or (event.xdata != None):
			ix, iy = int(event.ydata), int(event.xdata)

		if (str(event.inaxes).split('(')[0] == 'AxesSubplot') and button3.get_status()[0]:

			# default settings for levet set
			lambda1=1; lambda2=4; smoothing = 1; iterations = 100; rad = 3

			# advanced settings window pop-up
			figs, axx, slider_as1, slider_as2, slider_as3, slider_as4, slider_as5 = adv_window()
			figs.show()

			# to start segmentation
			input('Press enter to start segmentation: ')
			plt.close(figs)

			# add
			coords_zoom = (np.array([slider2.val, ix, iy])*np.array(zoom_fac)).astype(int)
			coords_orig = (int(slider2.val), int(ix), int(iy))
			print(coords_zoom,coords_orig)

			# apply level set to grow in 3D from single seed point
			seg_tmp = level_set3D(im_ct, coords_orig, list(reversed(res)), lambda1=slider_as1.val, lambda2=slider_as2.val, smoothing = int(slider_as3.val), iterations = int(slider_as4.val), rad = slider_as5.val)
			#seg_tmp = level_set3D(im_ct, coords_orig, list(reversed(res)), smoothing = int(slider_as3.val), iterations = int(slider_as4.val), rad = slider_as5.val, method = 'GAC', alpha = 150, sigma = 5, balloon = 1)

			# if no nodule was segmented, break; else continue
			if (seg_tmp is None):
				print('No nodule was segmented. Try changing parameters...')
				return None
			else:
				# because of interpolation, went from {0,1} -> [0,1]. Need to threshold to get binary segment
				seg_tmp[seg_tmp < 0.5] = 0
				seg_tmp[seg_tmp >= 0.5] = 1
				pred_st[seg_tmp == 1] = 1
				label_2d[seg_tmp == 1] = len(np.unique(label_2d)) # OBS: need to add new label for new segment, in order to remove it properly!

				seg_tmp_3d = zoom(seg_tmp.copy(), zoom = zoom_fac, order = 1)
				seg_tmp_3d[seg_tmp_3d < 0.5] = 0
				seg_tmp_3d[seg_tmp_3d >= 0.5] = 1
				pred_st_tmp[seg_tmp_3d == 1] = 1
				label_3d[seg_tmp_3d == 1] = len(np.unique(label_3d))

				# re-plot
				tmp_thr = pred_st[int(slider2.val)].copy()
				tmp_3d = pred_st_tmp.copy()
				threshold = slider1.val

				tmp_thr[tmp_thr < threshold] = 0
				tmp_thr[tmp_thr >= threshold] = 1

				tmp_3d[tmp_3d < threshold] = 0
				tmp_3d[tmp_3d >= threshold] = 1

				mlab.clf()
				mlab.contour3d(tmp_3d, colormap = 'hot', opacity = 1.0, vmin = 0, vmax = 1)
				if button1.get_status()[0]:
					mlab.contour3d(mask, colormap = 'Greys', opacity = 0.1, vmin = 0, vmax = 150)

				ax.clear()
				ax.imshow(im_ct[int(slider2.val), :, :], cmap = 'gray')
				#ax.imshow(tmp_thr, cmap = 'inferno', vmin = 0, vmax = 1, alpha = 0.3)
				ax.imshow(tmp_thr, cmap = 'gnuplot', vmin = 0, vmax = 2, alpha = 0.3)
				cursor = Cursor(ax, useblit=True, color='orange', linewidth=0.5)
				#ax.set_title('predicted')
				ax.set_axis_off()

				f.suptitle('slice '+str(slider2.val))
				f.canvas.draw_idle()


	def remove_mode(event):
		# if already in add mode -> switch to remove mode
		if button3.get_status()[0]:
			button3.set_active(0)

		f.canvas.mpl_connect('button_press_event', remove)


	def add_mode(event):
		# if already in remove mode -> switch to add mode
		if button2.get_status()[0]:
			button2.set_active(0)

		f.canvas.mpl_connect('button_press_event', add)


	def classify(event):
		if button4.get_status()[0]:
			pred_class = pred_VGG(im_ct, pred_st, res, VGG_model_path)	
			pred_class = zoom(pred_class.copy(), zoom = zoom_fac, order = 0)
			mlab.clf()
			print(np.unique(pred_class)[1:])

			for i in np.unique(pred_class):
				if i == 0:
					continue
				tmp = pred_class.copy()
				tmp[pred_class != i] = 0
				mlab.contour3d(tmp, colormap = 'OrRd', color = tuple(colors[int(round((9/5)*i)),:]), vmin = 1, vmax = 5)
				mlab.scalarbar(orientation = 'vertical', nb_labels = 9, label_fmt='%.1f')

			if button1.get_status()[0]:
				mlab.contour3d(mask, colormap = 'Greys', opacity = 0.1, vmin = 0, vmax = 150)

			mlab.orientation_axes(xlabel = 'z', ylabel = 'y', zlabel = 'x')

		elif not button4.get_status()[0]:
			# re-plot without classify
			tmp_thr = pred_st[int(slider2.val)].copy()
			tmp_3d = pred_st_tmp.copy()
			threshold = slider1.val

			tmp_thr[tmp_thr < threshold] = 0
			tmp_thr[tmp_thr >= threshold] = 1

			tmp_3d[tmp_3d < threshold] = 0
			tmp_3d[tmp_3d >= threshold] = 1

			mlab.clf()
			mlab.contour3d(tmp_3d, colormap = 'hot', opacity = 1.0, vmin = 0, vmax = 1)
			if button1.get_status()[0]:
				mlab.contour3d(mask, colormap = 'Greys', opacity = 0.1, vmin = 0, vmax = 150)

			ax.clear()
			ax.imshow(im_ct[int(slider2.val), :, :], cmap = 'gray')
			#ax.imshow(tmp_thr, cmap = 'inferno', vmin = 0, vmax = 1, alpha = 0.3)
			ax.imshow(tmp_thr, cmap = 'gnuplot', vmin = 0, vmax = 2, alpha = 0.3)
			ax.set_axis_off()

			f.suptitle('slice '+str(slider2.val))
			f.canvas.draw_idle()




	# Exit app when ESC is pressed
	def quit(event):
		if event.key == "escape":
			plt.close()
			mlab.close()



	# default settings for levet set for add-event
	lambda1=1; lambda2=4; smoothing = 1; iterations = 100; rad = 3


	# colormap for 3D-malignancy-plot for classify-event
	dz = list(range(1,10))
	norm = plt.Normalize()
	colors = plt.cm.OrRd(norm(dz))
	colors = np.array(colors)[:,:3]


	# plot to make simulator gator
	f, ax = plt.subplots(1,1, figsize = (12, 12))

	f.canvas.mpl_connect('key_press_event', up_scroll_alt)
	f.canvas.mpl_connect('key_press_event', down_scroll_alt)
	f.canvas.mpl_connect('scroll_event', up_scroll)
	f.canvas.mpl_connect('scroll_event', down_scroll)
	f.canvas.mpl_connect('key_press_event', quit)
	#f.canvas.mpl_connect('button_press_event', remove)
	#f.canvas.mpl_connect('button_press_event', add)

	b1ax = plt.axes([0.05, 0.2, 0.15, 0.11])
	b1ax.set_axis_off()
	b2ax = plt.axes([0.05, 0.35, 0.15, 0.11])
	b2ax.set_axis_off()
	b3ax = plt.axes([0.05, 0.5, 0.15, 0.11])
	b3ax.set_axis_off()
	b4ax = plt.axes([0.05, 0.65, 0.15, 0.11])
	b4ax.set_axis_off()


	s1ax = plt.axes([0.25, 0.08, 0.5, 0.03])
	s2ax = plt.axes([0.25, 0.02, 0.5, 0.03])

	slider1 = Slider(s1ax, 'threshold', 0.1, 1.0, dragging = True, valstep = 0.05)
	slider2 = Slider(s2ax, 'slice', 0.0, im_ct.shape[0]-1, valstep = 1)

	button1 = CheckButtons(b1ax, ['lung'], [False])
	button2 = CheckButtons(b2ax, ['Remove'], [False])
	button3 = CheckButtons(b3ax, ['Add'], [False])
	button4 = CheckButtons(b4ax, ['Classify'], [False])

	slider1.set_val(0.5)
	slider2.set_val(0)
	f.subplots_adjust(bottom = 0.15)

	#ax.imshow(im_ct[int(slider2.val), :, :], cmap = 'gray')
	#ax.imshow(pred_st[int(slider2.val), :, :], cmap = 'inferno', vmin = 0, vmax = 1, alpha = 0.3)
	ax.imshow(im_ct[int(slider2.val), :, :], cmap = 'gray')
	ax.imshow(pred_st[int(slider2.val), :, :], cmap = 'gnuplot', vmin = 0, vmax = 2, alpha = 0.3)
	cursor = Cursor(ax, useblit=True, color='orange', linewidth=0.5)
	ax.set_title('predicted')
	ax.set_axis_off()

	f.suptitle('slice '+str(slider2.val))

	button1.on_clicked(show_lung)
	button2.on_clicked(remove_mode)
	button3.on_clicked(add_mode)
	button4.on_clicked(classify)
	slider1.on_changed(new_thr)
	slider2.on_changed(images)

	plt.show()
	#mlab.show()




