(:action place-beside
    :parameters (?o - block ?u - block)
    :precondition (and (holding ?o) (on-table ?u))
    :effect (and (on-table ?o)
                 (beside ?o ?u)
                 (clear ?o)
                 (gripper-empty)
                 (not (holding ?o))))
