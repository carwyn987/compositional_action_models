(:action push-beside
    :parameters (?o - block ?u - block)
    :precondition (and (on-table ?o) (clear ?o) (on-table ?u) (gripper-empty) (not (beside ?o ?u)))
    :effect (and (beside ?o ?u)))
