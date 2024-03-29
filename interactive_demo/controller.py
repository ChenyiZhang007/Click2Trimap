import torch
import numpy as np
from tkinter import messagebox
import cv2
from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks
# from inference import single_inference, generator_tensor_dict
import time

def single_inference(image,trimap,model):

    trimap_nonp=trimap.copy()
    h,w,c=image.shape
    nonph,nonpw,_=image.shape
    newh= (((h-1)//32)+1)*32
    neww= (((w-1)//32)+1)*32
    padh=newh-h
    padh1=int(padh/2)
    padh2=padh-padh1
    padw=neww-w
    padw1=int(padw/2)
    padw2=padw-padw1
    image_pad=cv2.copyMakeBorder(image,padh1,padh2,padw1,padw2,cv2.BORDER_REFLECT)
    trimap_pad=cv2.copyMakeBorder(trimap,padh1,padh2,padw1,padw2,cv2.BORDER_REFLECT)
    h_pad,w_pad,_=image_pad.shape
    tritemp = np.zeros([*trimap_pad.shape, 3], np.float32)
    tritemp[:, :, 0] = (trimap_pad == 0)
    tritemp[:, :, 1] = (trimap_pad == 128)
    tritemp[:, :, 2] = (trimap_pad == 255)
    tritempimgs=np.transpose(tritemp,(2,0,1))
    tritempimgs=tritempimgs[np.newaxis,:,:,:]
    img=np.transpose(image_pad,(2,0,1))[np.newaxis,::-1,:,:]
    img=np.array(img,np.float32)
    img=img/255.
    img=torch.from_numpy(img).cuda()
    tritempimgs=torch.from_numpy(tritempimgs).cuda()

    with torch.no_grad():
        # time1 = time.time()
        pred=model(img,tritempimgs)
        # print('time_Matting:',time.time()-time1)
        pred=pred.detach().cpu().numpy()[0]
        pred=pred[:,padh1:padh1+h,padw1:padw1+w]
        preda=pred[0:1,]*255
        preda=np.transpose(preda,(1,2,0))
        preda=preda*(trimap_nonp[:,:,None]==128)+(trimap_nonp[:,:,None]==255)*255
    preda=np.array(preda,np.uint8)

    return preda

class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5):
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()

    def set_image(self, image):
        self.image = image
        self._result_mask = np.zeros_like(image, dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1

    def add_click(self, x, y, flag):
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })

        click = clicker.Click(flag=flag, coords=(y, x))
        self.clicker.add_click(click)
        # time1 = time.time()
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        # print('time_Clik2Trimap:',time.time()-time1)

        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)


        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        if not self.probs_history:
            self.reset_init_mask()
        self.update_image_callback()

    def change_background_1(self):
        # if not self.states:
        #     return

        # prev_state = self.states.pop()
        # self.clicker.set_state(prev_state['clicker'])
        # self.predictor.set_states(prev_state['predictor'])
        # # self.probs_history.pop()
        # if not self.probs_history:
        #     self.reset_init_mask()
        self.update_image_callback(backgroung_flag=1)

    def change_background_2(self):
        # if not self.states:
        #     return

        # prev_state = self.states.pop()
        # self.clicker.set_state(prev_state['clicker'])
        # self.predictor.set_states(prev_state['predictor'])
        # # self.probs_history.pop()
        # if not self.probs_history:
        #     self.reset_init_mask()
        self.update_image_callback(backgroung_flag=2)

    def change_background_3(self):
        # if not self.states:
        #     return

        # prev_state = self.states.pop()
        # self.clicker.set_state(prev_state['clicker'])
        # self.predictor.set_states(prev_state['predictor'])
        # # self.probs_history.pop()
        # if not self.probs_history:
        #     self.reset_init_mask()
        self.update_image_callback(backgroung_flag=3)

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]

            fore_mask = current_prob_additive.argmax(axis=0) == 2
            uk_mask = current_prob_additive.argmax(axis=0) == 1
            back_mask = current_prob_additive.argmax(axis=0) == 0


            pred_mask = np.stack([back_mask, uk_mask, fore_mask]).astype(np.uint8)
            return pred_mask
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            result_mask = result_mask.transpose(2,0,1) 
            # result_mask = np.
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
            return self.current_object_prob

    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        alpha = None

        results_mask_for_vis = self.result_mask
        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)

        if results_mask_for_vis is not None:
            trimap = results_mask_for_vis[0] * 0 + results_mask_for_vis[1] * 128 + results_mask_for_vis[2] * 255


        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)



            alpha = single_inference(self.image, trimap, self.matting_model)
            

            self.alpha_save = alpha
            self.trimap_save = trimap

        return vis, alpha, self.image
