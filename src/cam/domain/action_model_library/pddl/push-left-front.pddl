(:action push-left-front
    :parameters (?o - block)
    :precondition (and (on-table ?o) (clear ?o) (gripper-empty) (not (at-left ?o)) (not (at-front ?o)))
    :effect (and (at-left ?o) (at-front ?o)))
