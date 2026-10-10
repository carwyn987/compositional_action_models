(:action unstack
    :parameters (?o - block ?u - block)
    :precondition (and (on ?o ?u) (clear ?o) (gripper-empty))
    :effect (and (holding ?o)
                 (clear ?u)
                 (not (on ?o ?u))
                 (not (clear ?o))
                 (not (gripper-empty))))
