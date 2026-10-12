(:action push-left
    :parameters (?o - block)
    :precondition (and (on-table ?o) (clear ?o) (gripper-empty) (not (at-left ?o)))
    :effect (and (at-left ?o)))
