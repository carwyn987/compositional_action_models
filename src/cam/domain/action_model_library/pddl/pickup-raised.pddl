(:action pickup-raised
    :parameters (?o - block)
    :precondition (and (on-table ?o) (clear ?o) (gripper-empty))
    :effect (and (holding ?o)
                 (raised ?o)
                 (not (on-table ?o))
                 (not (clear ?o))
                 (not (gripper-empty))))
