import pygame
from pygame.locals import *
import numpy as np
import time
import math
import random

screenrect     = Rect(0, 0, 800, 800)
pygame.init()
try:
    pygame.mixer.init()
except pygame.error:
    pass

debug = True
# Set the display mode
winstyle = 0  # |FULLSCREEN
bestdepth = pygame.display.mode_ok(screenrect.size, winstyle, 32)
screen = pygame.display.set_mode(screenrect.size, winstyle, bestdepth)
entities = []

size = 20

cx,cy,cz = size/2.0, size/2.0, size*1.5
cvz = 0.0
clon, casi = 0.0, 0.0
block_colors = [[(100,0,0), (100,100,0), (0,100,0), (0,100,100), (0,0,100), (100,0,100)],
                [(200,0,0), (200,150,0), (0,200,0), (0,200,150), (0,0,200), (200,0,200)],
                [(255,0,0), (255,255,0), (0,255,0), (0,255,255), (0,0,255), (255,0,255)]]

def roll(array,diff,axis):
    out = np.roll(array,diff,axis)
    

def make_pos_list(x,y,z):
    return np.concatenate((x[...,None], y[...,None], z[...,None]),axis = 3).astype(int)

def terrain(x,y,z):
    out = np.zeros(x.shape,dtype = 'uint8')
    for i in range(3):
        k1 = random.randint(1,5)/20.
        k2 = random.randint(1,5)/20.
        wave = z < np.cos(x*k1) * np.cos(y*k2) * 5 + 10
        out[wave] = i+1
    out[...,0] = 0
    return out

pos_list = np.fromfunction(make_pos_list, (size,size,size))
face_points = np.array([[[0,0,0], [0,1,0], [0,1,1], [0,0,1]],
                        [[0,0,0], [1,0,0], [1,0,1], [0,0,1]],
                        [[0,0,0], [0,1,0], [1,1,0], [1,0,0]]])

class Entity():
    
    def __init__(self, x, y, z, hitbox):
        self.x, self.y, self.z = x,y,z
        self.vx, self.vy, self.vz = 0,0,0
        self.box = hitbox
        entities.append(self)
        self.consts = {(0,0,1):False, (0,1,0):False, (1,0,0):False, (0,0,-1):False, (0,-1,0):False, (-1,0,0):False}
    
    def move(self,x,y,z):
        pos = self.x, self.y, self.z
        change = x,y,z
        vel = self.vx, self.vy, self.vz
        do_drag = False
        for const in self.consts:
            if self.consts[const]:
                do_drag = True
                component = change[0]*const[0], change[1]*const[1], change[2]*const[2]
                if sum(component) > 0:
                    change = change[0]-component[0]*const[0], change[1]-component[1]*const[1], change[2]-component[2]*const[2]
                component = vel[0]*const[0], vel[1]*const[1], vel[2]*const[2]
                if sum(component) > 0:
                    vel = vel[0]-component[0]*const[0], vel[1]-component[1]*const[1], vel[2]-component[2]*const[2]
        self.x, self.y, self.z = pos[0] + change[0], pos[1] + change[1], pos[2] + change[2]
        self.vx, self.vy, self.vz = vel
        if do_drag:
            self.drag()
        
    
    def calc_constraints(self):
        axis_gap = 0.02
        
        new_pos = [self.x, self.y, self.z]
        #this system works by creating a slice of blocks that define each side of collision. the sides all share either the most positive point or the most negative point
        #the shlice is tested to see if any blocks are int it (2) and if there are blocks, the players location is rounded to outside the block (3)
        #axis gap is the system to make sure a side detects the collision of blocks head on before the side detectors sense the blocks. Each slice is moved out from the player by axis_gap
        for axis in range(3):
            
            coords = [(0,1,1), (1,0,1), (1,1,0)][axis]
            dir = [(1,0,0), (0,1,0), (0,0,1)][axis]
            
            # pos is the positive point of the hitbox rounded up to the nearest block. neg is the negative most point rounded down
            pos = [ int(math.ceil(self.x + axis_gap*dir[0])), int(math.ceil(self.y + axis_gap*dir[1])), int(math.ceil(self.z + axis_gap*dir[2])) ]
            neg = [ int(math.floor(self.x - self.box[0] - axis_gap*dir[0])), int(math.floor(self.y - self.box[1] -axis_gap * dir[1] )), int(math.floor(self.z - self.box[2] - axis_gap * dir[2])) +1]
            
            
            #positive side
            #pos 2 is the other point for the face
            pos2 = pos[0]*dir[0] + neg[0]*coords[0], pos[1]*dir[1] + neg[1]*coords[1], pos[2]*dir[2] + neg[2]*coords[2]
            blocks = world[pos2[0]:pos[0]+1, pos2[1]:pos[1]+1, pos2[2]:pos[2]+1]
            constraint = np.any(blocks != 0)
            self.consts[dir] = constraint #(2)
            if constraint:
                new_pos[axis] = math.floor(new_pos[axis] + axis_gap) - axis_gap #(3)
            #negititve side
            if axis == 2:
                neg[2] -= 1
            neg2 = neg[0]*dir[0] + pos[0]*coords[0], neg[1]*dir[1] + pos[1]*coords[1], neg[2]*dir[2] + pos[2]*coords[2]
            
            blocks = world[neg[0]:neg2[0]+1, neg[1]:neg2[1]+1, neg[2]:neg2[2]+1]
            constraint = np.any(blocks != 0)
            self.consts[(-dir[0],-dir[1],-dir[2])] = constraint
            if constraint:
                if axis == 2:
                    higher_blocks = world[neg[0]:neg2[0]+1, neg[1]:neg2[1]+1, neg[2]+1:neg2[2]+2]
                    if np.any(higher_blocks != 0):
                        new_pos[axis] = math.ceil(new_pos[axis] - self.box[axis] + axis_gap) + self.box[axis] + axis_gap
                else:
                    new_pos[axis] = math.ceil(new_pos[axis] - self.box[axis] - axis_gap) + self.box[axis] + axis_gap
                
        self.x, self.y, self.z = new_pos
    
    def drag(self):
        #exit()
        multiplier = 0.96
        self.vx *= multiplier
        self.vy *= multiplier
        self.vz *= multiplier
    
    def accel(self,x,y,z):
        self.vx += x
        self.vy += y
        self.vz += z
    
    def kill(self):
        entites.remove(self)
    
    def update(self):
        self.calc_constraints()
        self.accel(0,0,-0.01)
        self.move(self.vx, self.vy, self.vz)
        if (self.z < -20):
            self.kill()
    
class Player(Entity):
    
    def __init__(self,x,y,z):
        global cx,cy,cz
        Entity.__init__(self,x,y,z,(1.2,1.2,3))
        cx,cy,cz = x,y,z
        self.jump_bool = False
    
    def update(self):
        global cx, cy, cz
        keystate = pygame.key.get_pressed()
        multiplier = 0.01
        if keystate[119]: # w
            self.vy += multiplier * math.cos(clon)
            self.vx -= multiplier * math.sin(clon)
        if keystate[97]: # a
            self.vx -= multiplier * math.cos(clon)
            self.vy -= multiplier * math.sin(clon)
        if keystate[115]: # s
            self.vy -= multiplier * math.cos(clon)
            self.vx += multiplier * math.sin(clon)
        if keystate[100]: # d
            self.vx += multiplier * math.cos(clon)
            self.vy += multiplier * math.sin(clon)
        if keystate[32] and self.consts[(0,0,-1)]: # space
            self.vz += 0.2
        if keystate[304]: # shift
            self.vz -= multiplier
        Entity.update(self)
        cx,cy,cz = self.x, self.y, self.z
    
    def kill(self):
        exit()

class Chunk():
    
    def __init__(self,x,y,z):
        self.blocks = np.zeros((size,size,size))#np.fromfunction(terrain, (size,size,size))
        self.blocks[:,:,1:3] = (np.random.random((size,size,2))*3).astype('uint8')
        self.x, self.y, self.z = x,y,z
        self.faces, self.numbers, self.colors = None,None,None
        self.update()
    
    def update(self):
        rendermap = np.zeros((self.blocks.shape[0],self.blocks.shape[1],self.blocks.shape[2],3),dtype = 'uint8')
        blocks = self.blocks != 0
        
        #taking in the 3d block array and creating the rendermap
        #the rendermap has three numbers for each cube, one for each positive facing face on the cube
        #the int 
        for axis in range(3):
            sides = blocks.astype('int8') - np.roll(blocks,-1,axis) # suntract shifted bool mask to find places where ther is a change from air to block
            bool = sides > 0
            rendermap[...,axis][bool] = self.blocks[bool] # copying materials from chunk to rendermap
            #print chunk[sides > 0]
            bool = sides < 0
            rendermap[...,axis][bool] = np.roll(self.blocks,-1,axis)[bool] #chunk[np.roll(bool,1,axis)]
        
        mark0 = time.time()
        
        rendermap = rendermap.reshape((-1,3))
        poses = pos_list.copy().reshape((-1,3))
        poses[...,0] += self.x*size
        poses[...,1] += self.y*size
        poses[...,2] += self.z*size
        assert rendermap.shape == poses.shape
        keep = ~np.all(rendermap == 0, axis=1)
        rendermap = rendermap[keep]
        poses = poses[keep]
        #rendermap = np.concatenate((rendermap, pos_list), axis = 3)
        #quadlist = rendermap.reshape((-1,6))
        #rendermap = None
        #quadlist = quadlist[np.sum(quadlist[...,0:3] == 0,axis = 1) != 3]
        #poses = quadlist[...,3:]
        face_colors = rendermap.flatten()
        #quadlist = None
        numbers = np.repeat(np.array([[0],[1],[2]]),face_colors.shape[0]/3,axis=1).flatten('F') # axis numbers for each face
        #trimming out 0s
        bool = face_colors != 0
        face_colors = face_colors[bool]
        numbers = numbers[bool]
        poses = np.repeat(poses,3,axis=0)[bool]
        
        mark1 = time.time()
        
        #creating the faces
        N = poses.shape[0]
        faces = -face_points[numbers] + np.repeat(poses,4,axis=0).reshape((-1,4,3))
        
        
        
        assert poses.shape == ( N, 3 )
        assert numbers.shape == ( N, )
        assert face_colors.shape == ( N , ) and face_colors.dtype == 'uint8'
        assert faces.shape == ( N, 4, 3 )
        #print faces
        
        self.numbers = numbers
        self.faces = faces
        self.colors = face_colors
        self.N = N


class World():
    
    def __init__(self):
        self.chunks = {}
        for x in range(-2,2):
            for y in range(-2,2):
                for z in range(1):
                    self.chunks[(x,y,z)] = Chunk(x,y,z)
    
    def render(self):
        mark1 = time.time()
        N = 0
        for chunk in self.chunks.values():
            N += chunk.N
        faces = np.zeros((N,4,3))
        colors = np.zeros((N),dtype='uint8')
        numbers = np.zeros((N),dtype = int)
        N = 0
        for chunk in self.chunks.values():
            faces[N:N+chunk.N,...] = chunk.faces
            colors[N:N+chunk.N] = chunk.colors
            numbers[N:N+chunk.N] = chunk.numbers
            N += chunk.N
        
        x = faces[...,0].astype(float)
        y = faces[...,1].astype(float)
        z = faces[...,2].astype(float)
        x -= cx
        y -= cy
        z -= cz
        distance = ( (np.sum(x, axis=1)/4)**2 + (np.sum(y, axis=1)/4)**2 + (np.sum(z, axis = 1)/4)**2 ) ** 0.5
        index_array = np.argsort(distance)
        """ this is matrix multiplication    
          | cos(angle)  -sin(angle) |
          | sin(angle)   cos(angle) |
        """
        x2 = np.cos(-clon)*x - np.sin(-clon)*y
        y2 = np.sin(-clon)*x + np.cos(-clon)*y
        
        y3 = np.cos(-casi)*y2 - np.sin(-casi)*z
        z2 = np.sin(-casi)*y2 + np.cos(-casi)*z
        
        x,y,z = x2,y3,z2
        
        sx = np.pi/2 - np.arctan2(y,x)
        sy = - ( np.pi/2 - np.arctan2(y,z) )
        sx[sx > np.pi/2] = -np.pi/2
        sy[sy < -np.pi/2] = np.pi/2
        #print 'sx range:', np.min(sx), np.max(sx),'\t',
        #print 'sy range:', np.min(sy), np.max(sy)
        
        #sx = x/y
        #sy = -z/y
        points = np.stack((sx,sy),axis = 2)
        #points[points == np.nan] = 700
        #points[points == np.inf] = 700
        points *= screenrect.size[0]/(np.pi/2)
        points += screenrect.size[0]//2
        out_bounds = np.sum((points < 0) | (points > screenrect.size[0]), axis = (1,2)) == 8
        #areas = points[...,0,0]*(points[...,1,1]-points[...,2,1]) + points[...,1,0]*(points[...,1,
        new_index = np.arange(N)
        new_index[out_bounds] = -1
        #new_index[distance > 5] = -1
        index_array = new_index[index_array]
        index_array = index_array[index_array != -1]
        points  = points.astype(int)
        mark2 = time.time()
        
        #screen = np.zeros(screenrect.size, dtype='uint8')
        
                
        for i in index_array[::-1]:
            pygame.draw.polygon(screen,block_colors[numbers[i]][colors[i]], points[i,...].tolist(),0)
        
        print(mark2-mark1, time.time()-mark2)
    
    def generate_chunk(self,x,y,z):
        self.chunks[(x,y,z)] = Chunk(x,y,z)
    
    def __getitem__(self,a):
        chx, x = self.convert(a[0])
        chy, y = self.convert(a[1])
        chz, z = self.convert(a[2])
        chunk, low_pos = self.getchunk(chx, chy, chz)
        x = self.translate(x,-low_pos[0]*size)
        y = self.translate(y,-low_pos[1]*size)
        z = self.translate(z,-low_pos[2]*size)
        return chunk[x,y,z]
    
    def translate(self, x, value):
        if type(x) == int:
            return x+value
        elif type(x) == slice:
            return slice(x.start+value, x.stop+value)
        else:
            return x
    
    def getchunk(self,x,y,z):
        arrays = []
        x_range = range(x,x+1) if type(x) == int else range(x.start, x.stop)
        y_range = range(y,y+1) if type(y) == int else range(y.start, y.stop)
        z_range = range(z,z+1) if type(z) == int else range(z.start, z.stop)
        low_pos = [x_range[0], y_range[0], z_range[0]]
        full_array = np.zeros((size*len(x_range), size*len(y_range), size*len(z_range)), dtype = 'uint8')
        #nx > new x, chx > chunck x
        for nx,chx in enumerate(x_range):
            for ny,chy in enumerate(y_range):
                for nz,chz in enumerate(z_range):
                    if not (chx, chy, chz) in self.chunks:
                        self.generate_chunk(chx, chy, chz)
                    full_array[nx*size:(nx+1)*size, ny*size:(ny+1)*size, nz*size:(nz+1)*size ] = self.chunks[(chx,chy,chz)].blocks
        return full_array, low_pos
    
    def convert(self,item):
        if type(item) == int:
            return item//size, item
        elif type(item) == slice:
            start = self.convert(item.start)
            stop = self.convert(item.stop)
            return slice(start[0], stop[0]+1), slice(start[1], stop[1])
        else:
            return item,item
        
  
    #print mark0-start, mark1-mark0, mark2-mark1, mark3-mark2, mark3-start

world = World()

playing = True



#world = np.fromfunction(terrain,(size,size,size))

#island = (np.random.random((size/2,size/2,size/4))*6).astype('uint8')
# world = np.zeros((size,size,size),dtype = 'uint8')
# world[5:25, 5:25, 1:4] = 2
# world[5:10, 5:25, 4] = 3
# world[5:9, 5:25, 5] = 3
# world[5:8, 5:25, 6] = 3
# world[5:7, 5:25, 7] = 3
# world[5:6, 5:25, 8] = 3
# world[5:25, 5:10, 4:7] = 3
#print world

player = Player(8,8,15)

pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

clock = pygame.time.Clock()
while playing:
    
    pygame.draw.rect(screen, (130,220,255), screenrect)
    #render(world)
    
    world.render()
    
    pygame.display.flip()
    
        
    for event in pygame.event.get():
        #print event
        if event.type == KEYDOWN:
            #print event.key
            print( cx,cy,cz )
            print( clon, casi )
            pass
        if event.type == 5: # click
            pass
        if event.type == QUIT:
            playing = False
            break
        if (event.type == KEYDOWN and event.key == K_ESCAPE):
            playing = False
            break
        if event.type == pygame.MOUSEMOTION:
            pass 
    player.update()
    
    rel = pygame.mouse.get_rel()
    clon += -rel[0]/1000.
    casi += -rel[1]/1000.
    
    clock.tick(60)
    print( clock.get_fps() )
    