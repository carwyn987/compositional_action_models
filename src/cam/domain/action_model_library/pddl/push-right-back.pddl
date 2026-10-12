(:action push-right-back
    :parameters (?o - block)
    :precondition (and (on-table ?o) (clear ?o) (gripper-empty) (not (at-right ?o)) (not (at-back ?o)))
    :effect (and (at-right ?o) (at-back ?o)))
