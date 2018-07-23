;;; first painting problem

(define (problem painting001)
    (:domain painting)
    (:objects
        block

        ;; sprayer
        green-sprayer
        red-sprayer

        ;; colors
        green
        red
    )
    (:init

        ;; block stuff
        (clear block)
        (on-table block)
        (pickupable block)

        (clear green-sprayer)
        (on-table green-sprayer)
        (is-sprayer green-sprayer)
        (pickupable green-sprayer)

        (clear red-sprayer)
        (on-table red-sprayer)
        (is-sprayer red-sprayer)
        (pickupable red-sprayer)

        ;; spray coloring
        (color green-sprayer green)
        (color red-sprayer red)

        ;; arms
        (arm-empty)
    )
    (:goal
        (and
            (color block green)
            (arm-empty)
        )
    )
)