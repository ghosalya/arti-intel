;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Painting blocks world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain painting)
  (:requirements :strips)
  (:predicates (on ?x ?y)
			   (on-table ?x)
			   (clear ?x)
			   (arm-empty)
			   (holding ?x)
			   (color ?x ?rgb) ;;; added color
			   (is-sprayer ?x) ;;; check if box is sprayer
			   (pickupable ?x) ;;; check if pickupable
	       	)
  (:action pick-up
	     :parameters (?ob1)
	     :precondition (and 
		 					(clear ?ob1) 
							(on-table ?ob1) 
							(pickupable ?ob1)
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
	     :parameters (?sprayer ?targetbox ?targetcolor)
	     :precondition (and 
		 					(clear ?targetbox) 
							(on-table ?targetbox) 
							(holding ?sprayer) 
							(color ?sprayer ?targetcolor) 
							(is-sprayer ?sprayer)
						)
	     :effect
	     (and (color ?targetbox ?targetcolor))))

