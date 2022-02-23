import time
import cv2

from helpers import minutes, Logger

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

# Config
show_video = True
debug = True
frame_rate = 10
# Set these values much higher for production use
eye_confidence_seconds = 5
time_before_break = minutes(0.5)
break_time = minutes(0.5)

class EyeTracker:
    def __init__(self, show_video, debug, frame_rate, eye_confidence_seconds, time_before_break, break_time):
        self.show_video = show_video
        self.logger = Logger(debug)
        self.frame_rate = frame_rate
        self.eye_confidence_seconds = eye_confidence_seconds
        self.time_before_break = time_before_break
        self.break_time = break_time
        self.last_break = time.time()
        self.in_break = False
        self.need_break = False
        self.last_visibility_change = time.time()
        self.eyes_present = False
        self.pending_change = False

    def find_eyes(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        found_faces = len(faces) > 0
        found_eyes = False
        # if not found_faces:
        #     self.logger.log("Nothing to see here...")
        for (x,y,w,h) in faces:
            if show_video:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                found_eyes = True # Set this explicitly so there's no way to "un-see" eyes within a single frame if a face is found without eyes
            # if found_eyes:
            #     self.logger.log("Found eyes! - " + str(len(eyes)))
            # elif debug:
            #     self.logger.log("No eyes to be found")
            if show_video:
                for (ex,ey,ew,eh) in eyes:
                    color = (0,255,0)
                    if self.need_break:
                        color = (0,0,255)
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),color,2)    
        return found_eyes

    def check_eye_presence(self, eyes_in_frame):
        current_time = time.time()
        state_changed = False
        if eyes_in_frame != self.eyes_present:
            if self.pending_change and (current_time - self.last_visibility_change) > self.eye_confidence_seconds:
                self.logger.log("Eye visibility changed to " + str(eyes_in_frame))
                self.eyes_present = eyes_in_frame                
                state_changed = True
                self.pending_change = False
            elif self.pending_change:
                self.logger.log("Eyes might have changed visibility to " + str(eyes_in_frame) + " waiting for confirmation...")
            else:
                self.logger.log("Starting pending change")
                self.pending_change = True
                self.last_visibility_change = current_time
        elif self.pending_change:
            self.logger.log("Confirmation failed, keeping eye status = " + str(self.eyes_present))
            self.pending_change = False
        self.evaluate_break(current_time, state_changed)

    def evaluate_break(self, current_time, state_changed):
        if state_changed:
            state_change_time = current_time - self.eye_confidence_seconds # Un-offset the confidence timer            
            if not self.eyes_present and self.need_break and not self.in_break: # Eyes have gone away, probably to take a break so mark this time
                self.logger.log("Break started " + str(self.eye_confidence_seconds) + " seconds ago")
                self.in_break = True
                self.last_break = state_change_time
            elif self.eyes_present and self.in_break: # Eyes came back, let's see if they were gone long enough
                self.in_break = False
                if (state_change_time - self.last_break) >= self.break_time: # Good job
                    self.logger.log("Break was long enough, welcome back!")
                    self.need_break = False
                    self.last_break = current_time
                else:
                    self.logger.log("Break wasn't long enough! need_break will not be reset...")
        else:
            if self.eyes_present and not self.need_break and (current_time - self.last_break) > self.time_before_break:
                self.logger.log("Time for a break!")
                self.need_break = True
            elif not self.eyes_present and self.in_break and (current_time - self.last_break) >= self.break_time:
                self.logger.log("Break time was long enough, resetting status...")
                self.need_break = False

    def start(self):
        prev = 0
        while 1:
            time_elapsed = time.time() - prev
            res, img = cap.read()

            if time_elapsed > 1./frame_rate:
                prev = time.time()
                found_eyes = self.find_eyes(img)
                self.check_eye_presence(found_eyes)        
                    
                if show_video:
                    cv2.imshow('img',img)
            
            if show_video:        
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    eye_tracker = EyeTracker(show_video, debug, frame_rate, eye_confidence_seconds, time_before_break, break_time)
    eye_tracker.start()