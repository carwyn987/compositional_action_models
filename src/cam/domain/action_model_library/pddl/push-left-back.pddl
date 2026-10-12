(:action push-left-back
    :parameters (?o - block)
    :precondition (and (on-table ?o) (clear ?o) (gripper-empty) (not (at-left ?o)) (not (at-back ?o)))
    :effect (and (at-left ?o) (at-back ?o)))
