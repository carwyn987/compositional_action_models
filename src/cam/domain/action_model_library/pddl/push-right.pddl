(:action push-right
    :parameters (?o - block)
    :precondition (and (on-table ?o) (clear ?o) (gripper-empty) (not (at-right ?o)))
    :effect (and (at-right ?o)))
