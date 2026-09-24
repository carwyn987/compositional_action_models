(:action putdown
    :parameters (?o - block)
    :precondition (and (holding ?o))
    :effect (and (on-table ?o)
                 (clear ?o)
                 (gripper-empty)
                 (not (holding ?o))))
