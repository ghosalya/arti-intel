;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Painting blocks world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain painting)
  (:requirements :strips)
  (:predicates (on ?x ?y)
			   (on-table ?x)
			   (clear ?x)
			   (holding ?x)
			   (color ?x ?rgb) 
			   (can-pick ?x) 
			   (is-sprayer ?x)
			   (arm-empty))
  (:action pick-up
	     :parameters (?ob1)
	     :precondition (and 
		 					(clear ?ob1) 
							(on-table ?ob1) 
							(can-pick ?ob1)
							(arm-empty) 
						)
	     :effect
	     (and (not (on-table ?ob1))
		      (not (clear ?ob1))
		      (not (arm-empty))
		      (holding ?ob1)))
  (:action put-down
	     :parameters (?ob)
	     :precondition (holding ?ob)
	     :effect
	     (and (not (holding ?ob))
		   (clear ?ob)
		   (arm-empty)
		   (on-table ?ob)))
  (:action stack
	     :parameters (?sob ?sunderob)
	     :precondition (and (holding ?sob) (clear ?sunderob))
	     :effect
	     (and (not (holding ?sob))
		   (not (clear ?sunderob))
		   (clear ?sob)
		   (arm-empty)
		   (on ?sob ?sunderob)))
  (:action unstack
	     :parameters (?sob ?sunderob)
	     :precondition (and (on ?sob ?sunderob) (clear ?sob) (arm-empty))
	     :effect
	     (and (holding ?sob)
		   (clear ?sunderob)
		   (not (clear ?sob))
		   (not (arm-empty))
		   (not (on ?sob ?sunderob))))
   (:action spray
	     :parameters (?sprayer ?spraycolor ?box ?boxcolor)
	     :precondition (and 
		 					(clear ?box) 
							(on-table ?box) 
							(color ?box ?boxcolor) 
							(holding ?sprayer) 
							(color ?sprayer ?spraycolor) 
							(is-sprayer ?sprayer)
						)
	     :effect
	     (and 
		 	(color ?box ?spraycolor)
			(not (color ?box ?boxcolor))
			 )))

