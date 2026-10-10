(:action stack
    :parameters (?o - block ?u - block)
    :precondition (and (holding ?o) (clear ?u))
    :effect (and (on ?o ?u)
                 (clear ?o)
                 (gripper-empty)
                 (not (holding ?o))
                 (not (clear ?u))))
