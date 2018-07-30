;;; second painting problem

(define (problem painting002)
    (:domain painting)
    (:objects
        block-a 
        block-b

        ;; sprayer
        green-sprayer

        ;; colors
        green
        red
    )
    (:init
        ;; block stuff
        (clear block-a)
        (on-table block-a)
        (color block-a red)
        (can-pick block-a)

        (clear block-b)
        (on-table block-b)
        (color block-b red)
        (can-pick block-b)

        ;; sprayer stuff
        (clear green-sprayer)
        (on-table green-sprayer)
        (can-pick green-sprayer)
        (is-sprayer green-sprayer)
        (color green-sprayer green)

        (arm-empty)
    )
    (:goal
        (and
            (color block-b green)
            (on block-a block-b)
            (arm-empty)
        )
    )
)