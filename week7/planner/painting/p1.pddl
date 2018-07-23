;;; first painting problem

(define (problem painting001)
    (:domain painting)
    (:objects
        ;; block
        A 
        B

        ;; sprayer
        green-sprayer

        ;; colors
        green
        red
    )
    (:init
        (clear A)
        (on-table A)
        (color A red)

        (clear B)
        (on-table B)
        (color B red)

        (clear green-sprayer)
        (on-table green-sprayer)
        (color green-sprayer green)
    )
    (:goal
        (and
            (color B green)
            (on A B) ;; is this A on B?
            (arm-empty)
        )
    )
)