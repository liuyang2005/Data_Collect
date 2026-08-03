from r3kit.devices.gripper.xense.xense import Xense

gripper = Xense(id='d254505bfaaa', name='Xense')
print(gripper.read())
gripper.move(0.08)
