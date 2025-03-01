import './style.css'
import {regl} from './render'
import {update} from './interactive'


const tick = regl.frame(update)
