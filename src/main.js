import Clarity from '@microsoft/clarity';
const projectId = "qh7qxpa2do"
Clarity.init(projectId);

import './style.css'
import {regl} from './render'
import {update} from './interactive'


const tick = regl.frame(update)
