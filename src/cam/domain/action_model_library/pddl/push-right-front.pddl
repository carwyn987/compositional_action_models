(:action push-right-front
    :parameters (?o - block)
    :precondition (and (on-table ?o) (clear ?o) (gripper-empty) (not (at-right ?o)) (not (at-front ?o)))
    :effect (and (at-right ?o) (at-front ?o)))
