(:action pickup
    :parameters (?o - block)
    :precondition (and (on-table ?o) (clear ?o) (gripper-empty))
    :effect (and (holding ?o)
                 (not (on-table ?o))
                 (not (clear ?o))
                 (not (gripper-empty))))
