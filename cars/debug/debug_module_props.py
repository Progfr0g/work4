from core.module_props import ModuleProps
import core.module_names as module_names
import env

DEBUG_PROPS = ModuleProps(
    module_names.NUMBER_PLATES,
    env.TEST_CAMERA_ID,
    env.TEST_PARENT_SERVICE,
    database_host=env.TEST_DB_HOST,
    database_user=env.TEST_DB_USER,
    database_password=env.TEST_DB_PASSWORD,
    self_host="localhost",
    self_port=15001,
    camera_path=env.TEST_CAMERA_PATH
)
