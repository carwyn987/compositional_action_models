(:action push-front
    :parameters (?o - block)
    :precondition (and (on-table ?o) (clear ?o) (gripper-empty) (not (at-front ?o)))
    :effect (and (at-front ?o)))
