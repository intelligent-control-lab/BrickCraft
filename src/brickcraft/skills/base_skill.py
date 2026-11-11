from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple




class BaseSkill(ABC):
    """
    Abstract base class for all skills in the brick assembly pipeline.
    """

    def __init__(self):
        pass

    
    def run(self, skill_input: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        pre_condition_check -> execute skill -> post_condition

        Args:
            skill_input (Dict[str, Any]): Input data for the skill execution.
        Returns:
            bool: True if the skill executed successfully, False otherwise.
        """

        print(f"[{self.__class__.__name__}] Checking Pre-condition...")
        if not self.pre_condition(skill_input):
            print(f"[{self.__class__.__name__}] Pre-condition Failed.")
            return False, {}

        print(f"[{self.__class__.__name__}] Executing...")
        try:
            output = self.execute(skill_input) or {}
        except Exception as e:
            print(f"[{self.__class__.__name__}] Execution Error: {e}")
            return False, {}

        print(f"[{self.__class__.__name__}] Checking Post-condition...")
        if not self.post_condition(skill_input):
            print(f"[{self.__class__.__name__}] Post-condition Failed.")
            return False, output

        return True, output


    @abstractmethod
    def pre_condition(self, skill_input: Dict[str, Any]) -> bool:
        """
        Check if the pre-conditions for executing the skill are met.

        Returns:
            bool: True if pre-conditions are met, False otherwise.
        """
        raise NotImplementedError
    
    @abstractmethod
    def execute(self, skill_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the main logic of the skill.
        """
        raise NotImplementedError


    @abstractmethod
    def post_condition(self, skill_input: Dict[str, Any]) -> bool:
        """
        Check if the post-conditions after executing the skill are met.
        In other words, check whether the skill has succeeded.

        Returns:
            bool: True if post-conditions are met, False otherwise.
        """
        raise NotImplementedError









