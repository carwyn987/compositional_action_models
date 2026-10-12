(:action push-back
    :parameters (?o - block)
    :precondition (and (on-table ?o) (clear ?o) (gripper-empty) (not (at-back ?o)))
    :effect (and (at-back ?o)))
