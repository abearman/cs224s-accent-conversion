;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;;                    Language Technology Institute                    ;;;
;;;                     Carnegie Mellon University                      ;;;
;;;                         Copyright (c) 2003                          ;;;
;;;                        All Rights Reserved.                         ;;;
;;;                                                                     ;;;
;;; Permission is hereby granted, free of charge, to use and distribute ;;;
;;; this software and its documentation without restriction, including  ;;;
;;; without limitation the rights to use, copy, modify, merge, publish, ;;;
;;; distribute, sublicense, and/or sell copies of this work, and to     ;;;
;;; permit persons to whom this work is furnished to do so, subject to  ;;;
;;; the following conditions:                                           ;;;
;;;  1. The code must retain the above copyright notice, this list of   ;;;
;;;     conditions and the following disclaimer.                        ;;;
;;;  2. Any modifications must be clearly marked as such.               ;;;
;;;  3. Original authors' names are not deleted.                        ;;;
;;;  4. The authors' names are not used to endorse or promote products  ;;;
;;;     derived from this software without specific prior written       ;;;
;;;     permission.                                                     ;;;
;;;                                                                     ;;;
;;; CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK        ;;;
;;; DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING     ;;;
;;; ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT  ;;;
;;; SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE LIABLE     ;;;
;;; FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES   ;;;
;;; WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN  ;;;
;;; AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,         ;;;
;;; ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF      ;;;
;;; THIS SOFTWARE.                                                      ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;                                                                     ;;;
;;; Code for building F0 unit selection models                          ;;;
;;;                                                                     ;;;
;;; This file is only used at database build time                       ;;;
;;;                                                                     ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(require 'clunits_build)

;;; Basic voice definition file with voice defines and clunit
;;; parameter definition for run time.
(load "festvox/cmu_us_awb_arctic_f0clunits.scm")

;;; Add Build time parameters
(set! cmu_us_awb_arctic::f0_dt_params
      (cons
       ;; in case cmu_us_awb_arctic_f0clunits defines this too, put this at start
       (list 'db_dir (string-append cmu_us_awb_arctic::f0_clunits_dir "/"))
       (append
	cmu_us_awb_arctic::f0_dt_params
	(list
	;;; In cmu_us_awb_arctic_clunits.scm
	 ;;'(coeffs_dir "lpc/")
	 ;;'(coeffs_ext ".lpc")
	 '(disttabs_dir "festival/disttabs/")
	 '(utts_dir "festival/utts/")
	 '(utts_ext ".utt")
	 '(dur_pen_weight 0.0)
	 '(f0_pen_weight 0.0)
	 '(get_stds_per_unit t)
	 '(ac_left_context 0.8)
	 '(ac_weights
	   (0.5 0.5))
	 ;; Join weights in cmu_us_awb_arctic_f0clunits.scm
	 ;; Features for extraction
	 '(feats_dir "festival/f0feats/")
	 '(feats 
	   (occurid
	    p.name p.ph_vc p.ph_ctype 
	    p.ph_vheight p.ph_vlng 
	    p.ph_vfront  p.ph_vrnd 
	    p.ph_cplace  p.ph_cvox    
	    n.name n.ph_vc n.ph_ctype 
	    n.ph_vheight n.ph_vlng 
	    n.ph_vfront  n.ph_vrnd 
	    n.ph_cplace  n.ph_cvox
	    segment_duration 
	    seg_pitch p.seg_pitch n.seg_pitch
	    R:SylStructure.parent.stress 
	    seg_onsetcoda n.seg_onsetcoda p.seg_onsetcoda
	    R:SylStructure.parent.accented 
	    pos_in_syl 
	    syl_initial
	    syl_final
	    R:SylStructure.parent.syl_break 
	    R:SylStructure.parent.R:Syllable.p.syl_break
	    R:SylStructure.parent.position_type
	    pp.name pp.ph_vc pp.ph_ctype 
	    pp.ph_vheight pp.ph_vlng 
	    pp.ph_vfront  pp.ph_vrnd 
	    pp.ph_cplace pp.ph_cvox
            n.lisp_is_pau
            p.lisp_is_pau
	    R:SylStructure.parent.parent.gpos
	    R:SylStructure.parent.parent.R:Word.p.gpos
	    R:SylStructure.parent.parent.R:Word.n.gpos
	    ))
	 ;; Wagon tree building params
;	 (trees_dir "festvox/")  ;; in cmu_us_awb_arctic_f0clunits.scm
	 '(wagon_field_desc "festival/clunits/f0clunits.desc")
	 '(wagon_progname "$ESTDIR/bin/wagon")
	 '(wagon_cluster_size 20)
	 '(prune_reduce 0)
	 '(cluster_prune_limit 40)
	 ;; The dictionary of units used at run time
;	 (catalogue_dir "festvox/")   ;; in cmu_us_awb_arctic_f0clunits.scm
	 ;;  Run time parameters 
	 ;; all in cmu_us_awb_arctic_f0clunits.scm
	 ;; Files in db, filled in at build_f0clunits time
	 ;; (files ("time0001" "time0002" ....))
))))

(define (build_f0clunits file)
  "(build_clunits file)
Build cluster synthesizer for the given recorded data and domain."
  (build_f0clunits_init file)
  (do_all)  ;; someday I'll change the name of this function
)

(define (build_f0clunits_init file)
  "(build_clunits_init file)
Get setup ready for (do_all) (or (do_init))."
;  (eval (list cmu_us_awb_arctic::closest_voice))

  ;; Add specific fileids to the list for this run
  (set! cmu_us_awb_arctic::f0_dt_params
	(append
	 cmu_us_awb_arctic::f0_dt_params
	 (list
	  (list
	   'files
	   (mapcar car (load file t))))))
  
  (set! dt_params cmu_us_awb_arctic::f0_dt_params)
  (set! clunits_params cmu_us_awb_arctic::f0_dt_params)
)
