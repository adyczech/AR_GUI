from src.AR_Model import *


class ARCube:

    def __init__(self, path, cube_size, cube_ids):
        self.mesh = ARModel(path)
        self.cube_size = cube_size
        self.cube_ids = cube_ids  # [front, right, back, left, top, bottom]
        self.drawn = False

    def draw_cube_model(self, view_matrix, id, x, y, z, scale, rx, ry, rz):
        if not self.drawn:
            if self.cube_ids[0] == id:
                self.mesh.draw_model(view_matrix, x, y, -self.cube_size / 2 + z, scale, 0, 0, 0)
            elif self.cube_ids[1] == id:
                self.mesh.draw_model(view_matrix, x, y, -self.cube_size / 2, scale, 0, -90, 0)
            elif self.cube_ids[2] == id:
                self.mesh.draw_model(view_matrix, x, y, -self.cube_size / 2, scale, 0, -180, 0)
            elif self.cube_ids[3] == id:
                self.mesh.draw_model(view_matrix, x, y, -self.cube_size / 2, scale, 0, -270, 0)
            elif self.cube_ids[4] == id:
                self.mesh.draw_model(view_matrix, x, -z, y - self.cube_size / 2, scale, 90, 0, 0)
            elif self.cube_ids[5] == id:
                self.mesh.draw_model(view_matrix, x, z, -y - self.cube_size / 2, scale, -90, 0, 0)

            self.drawn = True
